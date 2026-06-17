"""Offline prediction metrics (Sections 13.6, 13.7, 13.10, 15.2).

Reports normalized MSE, cosine, R^2 (against the target mean), and — for matrix
targets — subspace overlap and future-energy capture. Skill scores compare a
model against each baseline (incremental predictability beyond strong
baselines).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..utils import linalg
import torch

EPS = 1e-12


def _masked(Y, mask):
    return Y * mask[None]


def mse(Yhat, Y, mask) -> float:
    d = (Yhat - Y) * mask[None]
    return float((d ** 2).sum() / (mask.sum() * Y.shape[0] + EPS))


def normalized_mse(Yhat, Y, mask) -> float:
    num = (((Yhat - Y) * mask[None]) ** 2).sum()
    den = ((Y * mask[None]) ** 2).sum() + EPS
    return float(num / den)


def cosine(Yhat, Y, mask) -> float:
    a = (Yhat * mask[None]).reshape(Yhat.shape[0], -1)
    b = (Y * mask[None]).reshape(Y.shape[0], -1)
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + EPS
    return float(np.mean(num / den))


def r2_vs_mean(Yhat, Y, mask) -> float:
    ybar = (Y * mask[None]).mean(0, keepdims=True)
    ss_res = (((Yhat - Y) * mask[None]) ** 2).sum()
    ss_tot = (((Y - ybar) * mask[None]) ** 2).sum() + EPS
    return float(1 - ss_res / ss_tot)


def skill_vs(Yhat, Y_baseline, Y, mask) -> float:
    """1 - MSE(model)/MSE(baseline): positive means the model beats baseline."""
    m_model = mse(Yhat, Y, mask)
    m_base = mse(Y_baseline, Y, mask) + EPS
    return float(1 - m_model / m_base)


def subspace_metrics(Yhat, Y, mask, node_shapes, rank=8) -> Dict[str, float]:
    """Mean subspace overlap and energy capture for matrix-valued targets."""
    N, nodes, Td = Y.shape
    overlaps, energies = [], []
    for ni in range(nodes):
        shape = tuple(int(s) for s in node_shapes[ni])
        flat = int(np.prod(shape))
        r = min(rank, shape[0], shape[1])
        for b in range(min(N, 64)):  # subsample for speed
            Gt = torch.from_numpy(Y[b, ni, :flat].reshape(shape))
            Gp = torch.from_numpy(Yhat[b, ni, :flat].reshape(shape))
            if Gt.norm() < EPS or Gp.norm() < EPS:
                continue
            Ut, _, Vt = linalg.top_singular(Gt, r)
            Up, _, Vp = linalg.top_singular(Gp, r)
            overlaps.append(linalg.subspace_overlap(Ut, Up))
            energies.append(linalg.energy_capture(Up, Gt, Vp))
    return {
        "subspace_overlap": float(np.mean(overlaps)) if overlaps else float("nan"),
        "energy_capture": float(np.mean(energies)) if energies else float("nan"),
    }


def basis_subspace_metrics(Yhat, Y, mask, node_shapes) -> Dict[str, float]:
    """For the ``gradient_subspace`` target the per-node values ARE orthonormal
    bases U (m x r); compare predicted vs true column spaces with the rotation/
    sign-invariant overlap and projector distance (Section 20.5)."""
    N, nodes, Td = Y.shape
    overlaps, dists = [], []
    for ni in range(nodes):
        shape = tuple(int(s) for s in node_shapes[ni])
        flat = int(np.prod(shape))
        for b in range(min(N, 64)):
            Ut = torch.from_numpy(Y[b, ni, :flat].reshape(shape))
            Up = torch.from_numpy(Yhat[b, ni, :flat].reshape(shape))
            if Ut.norm() < EPS or Up.norm() < EPS:
                continue
            Up = torch.linalg.qr(Up)[0]  # re-orthonormalize the prediction
            overlaps.append(linalg.subspace_overlap(Ut, Up))
            dists.append(linalg.projection_distance(Ut, Up))
    return {
        "subspace_overlap": float(np.mean(overlaps)) if overlaps else float("nan"),
        "projection_distance": float(np.mean(dists)) if dists else float("nan"),
    }


def evaluate_prediction(Yhat, Y, mask, node_shapes, target: str, baselines: Dict[str, np.ndarray] = None) -> dict:
    out = {
        "nmse": normalized_mse(Yhat, Y, mask),
        "cosine": cosine(Yhat, Y, mask),
        "r2": r2_vs_mean(Yhat, Y, mask),
        "mse": mse(Yhat, Y, mask),
    }
    is_matrix = target in ("future_gradient", "next_gradient", "future_update", "next_update",
                           "future_weight", "next_weight", "weight_delta", "future_weight_delta")
    if is_matrix:
        out.update(subspace_metrics(Yhat, Y, mask, node_shapes))
    elif target == "gradient_subspace":
        out.update(basis_subspace_metrics(Yhat, Y, mask, node_shapes))
    if baselines:
        for bname, bpred in baselines.items():
            out[f"skill_vs_{bname}"] = skill_vs(Yhat, bpred, Y, mask)
    return out
