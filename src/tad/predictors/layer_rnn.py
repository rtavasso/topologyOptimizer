"""Independent per-layer sequence predictor (Section 11.2).

A shared GRU runs over each layer's history independently (no cross-layer
information), giving the control that isolates the topology contribution.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import Predictor
from ._torch_util import train_module, predict_module


class _LayerGRUModule(nn.Module):
    def __init__(self, feat_dim, tgt_dim, hidden=128, layers=1):
        super().__init__()
        self.gru = nn.GRU(feat_dim, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, tgt_dim))

    def forward(self, feat, tgt_hist):
        # feat: (B, H, nodes, F)
        B, H, nodes, F = feat.shape
        x = feat.permute(0, 2, 1, 3).reshape(B * nodes, H, F)
        out, _ = self.gru(x)
        y = self.head(out[:, -1])
        return y.reshape(B, nodes, -1)


class LayerGRU(Predictor):
    name = "layer_gru"
    needs_training = True

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.hidden = int((cfg or {}).get("hidden", 128))
        self.epochs = int((cfg or {}).get("epochs", 60))
        self.lr = float((cfg or {}).get("lr", 1e-3))
        self.device = (cfg or {}).get("device", "cpu")
        self.module = None

    def fit(self, bundle):
        feat = bundle["train_feat"]
        tgt_dim = bundle["train_Y"].shape[-1]
        self.module = _LayerGRUModule(feat.shape[-1], tgt_dim, self.hidden)
        self.fwd = lambda m, f, t: m(f, t)
        train_module(self.module, bundle, self.fwd, epochs=self.epochs, lr=self.lr, device=self.device)

    def predict(self, bundle, split):
        return predict_module(self.module, bundle, split, self.fwd, self.device)
