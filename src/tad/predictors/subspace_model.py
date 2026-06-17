"""Slow-subspace / fast-coordinate model (Section 11.4, H3).

Models G_t ~ U_t C_t V_t^T where (U, V) evolve slowly. We estimate the slow
basis from an EMA of the historical gradients (SVD of the EMA), then take
Option B: derive the coefficient matrix from the most recent gradient,
C = U^T G_last V, and reconstruct U C V^T. This remains useful even when exact
future gradients are unpredictable (the bet of H3 / interpretation rule 6).
"""
from __future__ import annotations

import numpy as np

from .base import Predictor, split_arrays
from ..utils import linalg
import torch


class SlowSubspaceFastCoordinate(Predictor):
    name = "slow_subspace_fast_coordinate"

    def __init__(self, cfg=None):
        super().__init__(cfg)
        cfg = cfg or {}
        self.rank = int(cfg.get("rank", 8))
        self.beta = float(cfg.get("ema_beta", 0.9))

    def predict(self, bundle, split):
        _, tgt_hist, _ = split_arrays(bundle, split)
        N, H, nodes, Td = tgt_hist.shape
        shapes = bundle["node_shapes"]
        out = np.zeros((N, nodes, Td), dtype=np.float32)
        for ni in range(nodes):
            shape = tuple(int(s) for s in shapes[ni])
            flat = int(np.prod(shape))
            r = min(self.rank, shape[0], shape[1])
            # EMA of historical gradient matrices -> slow basis
            ema = tgt_hist[:, 0, ni, :flat]
            for t in range(1, H):
                ema = self.beta * ema + (1 - self.beta) * tgt_hist[:, t, ni, :flat]
            last = tgt_hist[:, -1, ni, :flat]
            for b in range(N):
                Gema = torch.from_numpy(ema[b].reshape(shape))
                Glast = torch.from_numpy(last[b].reshape(shape))
                U, S, V = linalg.top_singular(Gema, r)
                C = U.transpose(0, 1) @ Glast @ V
                rec = (U @ C @ V.transpose(0, 1)).reshape(-1).numpy()
                out[b, ni, :flat] = rec
        return out
