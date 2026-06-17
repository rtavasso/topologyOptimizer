"""Topology-aware graph dynamics model (Section 11.3).

Nodes are linear maps; edges are composition relationships. We run K_g message-
passing layers across nodes at each historical time, then a temporal GRU per
node, then a per-node decoder. The ``residual`` variant predicts the residual
beyond a tuned-EMA baseline (Section 12.10), so it cannot win merely by
rediscovering EMA. A ``shuffle_graph`` flag provides the Section-13.12 control.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .base import Predictor
from ._torch_util import train_module, predict_module


def _chain_adjacency(n: int) -> np.ndarray:
    A = np.eye(n, dtype=np.float32)
    for i in range(n - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    d = A.sum(1, keepdims=True)
    return A / np.clip(d, 1.0, None)


class _MessagePassing(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.edge = nn.Linear(dim, dim)
        self.node = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU())

    def forward(self, z, A):  # z: (B, nodes, dim), A: (nodes, nodes)
        agg = torch.einsum("ij,bjd->bid", A, self.edge(z))
        return self.node(torch.cat([z, agg], dim=-1))


class _GraphGRUModule(nn.Module):
    def __init__(self, feat_dim, tgt_dim, n_nodes, adjacency, hidden=128, kg=2,
                 residual=False, ema_beta=0.9):
        super().__init__()
        self.register_buffer("A", torch.from_numpy(adjacency))
        self.enc = nn.Sequential(nn.Linear(feat_dim, hidden), nn.GELU())
        self.mp = nn.ModuleList([_MessagePassing(hidden) for _ in range(kg)])
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.dec = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, tgt_dim))
        self.residual = residual
        self.ema_beta = ema_beta

    def _ema(self, tgt_hist):
        out = tgt_hist[:, 0]
        for t in range(1, tgt_hist.shape[1]):
            out = self.ema_beta * out + (1 - self.ema_beta) * tgt_hist[:, t]
        return out

    def forward(self, feat, tgt_hist):
        B, H, nodes, F = feat.shape
        z = self.enc(feat)                       # (B,H,nodes,hidden)
        z = z.reshape(B * H, nodes, -1)
        for mp in self.mp:
            z = z + mp(z, self.A)
        z = z.reshape(B, H, nodes, -1).permute(0, 2, 1, 3).reshape(B * nodes, H, -1)
        out, _ = self.gru(z)
        pred = self.dec(out[:, -1]).reshape(B, nodes, -1)
        if self.residual:
            pred = pred + self._ema(tgt_hist)
        return pred


class TopologyGraphGRU(Predictor):
    name = "topology_graph_gru"
    needs_training = True

    def __init__(self, cfg=None, residual=False):
        super().__init__(cfg)
        cfg = cfg or {}
        self.hidden = int(cfg.get("hidden", 128))
        self.kg = int(cfg.get("kg", 2))
        self.epochs = int(cfg.get("epochs", 60))
        self.lr = float(cfg.get("lr", 1e-3))
        self.device = cfg.get("device", "cpu")
        self.shuffle_graph = bool(cfg.get("shuffle_graph", False))
        self.residual = residual
        self.module = None

    def fit(self, bundle):
        feat = bundle["train_feat"]
        n_nodes = feat.shape[2]
        tgt_dim = bundle["train_Y"].shape[-1]
        A = _chain_adjacency(n_nodes)
        if self.shuffle_graph and n_nodes > 2:
            rng = np.random.default_rng(0)
            perm = rng.permutation(n_nodes)
            A = A[perm][:, perm]
        self.module = _GraphGRUModule(feat.shape[-1], tgt_dim, n_nodes, A,
                                      hidden=self.hidden, kg=self.kg, residual=self.residual)
        self.fwd = lambda m, f, t: m(f, t)
        train_module(self.module, bundle, self.fwd, epochs=self.epochs, lr=self.lr, device=self.device)

    def predict(self, bundle, split):
        return predict_module(self.module, bundle, split, self.fwd, self.device)
