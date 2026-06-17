"""Predictor training objectives (Section 12).

All losses are normalized where the spec normalizes, and each is logged
separately so its contribution can be tracked. ``combined_loss`` assembles the
default weighted objective (Section 12.11).
"""
from __future__ import annotations

from typing import Dict

import torch

EPS = 1e-8


def _nrm(num, den):
    return num / (den + EPS)


def parameter_loss(W_hat, W):  # 12.1
    return _nrm(torch.linalg.norm((W_hat - W).reshape(-1)) ** 2, torch.linalg.norm(W.reshape(-1)) ** 2)


def delta_loss(dW_hat, dW):  # 12.2
    return _nrm(torch.linalg.norm((dW_hat - dW).reshape(-1)) ** 2, torch.linalg.norm(dW.reshape(-1)) ** 2)


def direction_loss(dW_hat, dW):  # 12.3
    a = dW_hat.reshape(-1)
    b = dW.reshape(-1)
    return 1 - (a @ b) / (a.norm() * b.norm() + EPS)


def probe_loss(WP_hat, WP):  # 12.4
    return _nrm(torch.linalg.norm((WP_hat - WP).reshape(-1)) ** 2, torch.linalg.norm(WP.reshape(-1)) ** 2)


def end_to_end_map_loss(M_hat, M):  # 12.5
    return _nrm(torch.linalg.norm((M_hat - M).reshape(-1)) ** 2, torch.linalg.norm(M.reshape(-1)) ** 2)


def functional_loss(f_hat, f):  # 12.6
    return _nrm(torch.linalg.norm((f_hat - f).reshape(-1)) ** 2, torch.linalg.norm(f.reshape(-1)) ** 2)


def subspace_loss(U_hat, U):  # 12.7
    P1 = U_hat @ U_hat.transpose(-2, -1)
    P2 = U @ U.transpose(-2, -1)
    return torch.linalg.norm((P1 - P2).reshape(-1)) ** 2


def spectral_loss(sigma_hat, sigma):  # 12.8
    return torch.linalg.norm(torch.log(sigma_hat + EPS) - torch.log(sigma + EPS)) ** 2


def rollout_loss(pred_seq, true_seq, gamma=0.9):  # 12.9
    total = 0.0
    for k, (sh, st) in enumerate(zip(pred_seq, true_seq)):
        total = total + (gamma ** k) * torch.linalg.norm((sh - st).reshape(-1)) ** 2
    return total


def residualized_loss(R_hat, Y, baseline):  # 12.10
    R = Y - baseline
    return _nrm(torch.linalg.norm((R_hat - R).reshape(-1)) ** 2, torch.linalg.norm(R.reshape(-1)) ** 2)


def combined_loss(components: Dict[str, torch.Tensor], weights: Dict[str, float]) -> torch.Tensor:
    total = 0.0
    for k, v in components.items():
        total = total + weights.get(k, 1.0) * v
    return total
