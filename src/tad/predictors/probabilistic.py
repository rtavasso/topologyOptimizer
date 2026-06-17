"""Probabilistic predictor (Section 11.5).

Conditional Gaussian over the target with a predicted mean and diagonal
variance, trained by Gaussian NLL. Calibration of the predicted variance is
reported in offline metrics. Used only after deterministic baselines
(interpretation rule: deterministic first); a diffusion variant is intentionally
deferred until multimodality is demonstrated.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .base import Predictor, split_arrays


class _GaussianModule(nn.Module):
    def __init__(self, feat_dim, tgt_dim, hidden=128):
        super().__init__()
        self.gru = nn.GRU(feat_dim, hidden, batch_first=True)
        self.mean = nn.Linear(hidden, tgt_dim)
        self.logvar = nn.Linear(hidden, tgt_dim)

    def forward(self, feat):
        B, H, nodes, F = feat.shape
        x = feat.permute(0, 2, 1, 3).reshape(B * nodes, H, F)
        out, _ = self.gru(x)
        h = out[:, -1]
        mean = self.mean(h).reshape(B, nodes, -1)
        logvar = self.logvar(h).reshape(B, nodes, -1).clamp(-8, 8)
        return mean, logvar


class ConditionalGaussian(Predictor):
    name = "conditional_gaussian"
    needs_training = True

    def __init__(self, cfg=None):
        super().__init__(cfg)
        cfg = cfg or {}
        self.hidden = int(cfg.get("hidden", 128))
        self.epochs = int(cfg.get("epochs", 60))
        self.lr = float(cfg.get("lr", 1e-3))
        self.device = cfg.get("device", "cpu")
        self.module = None
        self.last_logvar = None

    def fit(self, bundle):
        feat = bundle["train_feat"]
        tgt_dim = bundle["train_Y"].shape[-1]
        self.module = _GaussianModule(feat.shape[-1], tgt_dim, self.hidden).to(self.device)
        mask = torch.from_numpy(bundle["mask"]).float().to(self.device)
        feat_t = torch.from_numpy(feat).float().to(self.device)
        Y_t = torch.from_numpy(bundle["train_Y"]).float().to(self.device)
        opt = torch.optim.AdamW(self.module.parameters(), lr=self.lr, weight_decay=1e-5)
        n = feat_t.shape[0]
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, 256):
                idx = perm[i:i + 256]
                opt.zero_grad()
                mean, logvar = self.module(feat_t[idx])
                m = mask.unsqueeze(0)
                nll = 0.5 * (logvar + (Y_t[idx] - mean) ** 2 / logvar.exp()) * m
                loss = nll.sum() / (m.sum() * len(idx) + 1e-8)
                loss.backward()
                opt.step()

    def predict(self, bundle, split):
        feat, _, _ = split_arrays(bundle, split)
        self.module.eval()
        with torch.no_grad():
            mean, logvar = self.module(torch.from_numpy(feat).float().to(self.device))
        self.last_logvar = logvar.cpu().numpy()
        return mean.cpu().numpy()
