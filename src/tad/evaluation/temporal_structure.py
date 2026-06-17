"""Temporal-structure analyses on a raw trajectory (Section 13).

These are mandatory before optimizer integration. Everything here operates
directly on a logged trajectory (no learned model) and produces the curves the
first report requires (Section 22 items 2-8, 11-12).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..logging.reader import TrajectoryReader
from ..utils import linalg
import torch

PHASES = [(0.0, 0.05), (0.05, 0.20), (0.20, 0.60), (0.60, 0.90), (0.90, 1.0001)]
PHASE_NAMES = ["0-5%", "5-20%", "20-60%", "60-90%", "90-100%"]


def _cos(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def autocorrelation(series: np.ndarray, horizons: List[int]) -> Dict[int, float]:
    s = np.asarray(series, dtype=np.float64).reshape(len(series), -1)
    s = s - s.mean(0, keepdims=True)
    out = {}
    for h in horizons:
        if h >= len(s):
            out[h] = float("nan")
            continue
        a, b = s[:-h], s[h:]
        num = (a * b).sum()
        den = np.sqrt((a ** 2).sum() * (b ** 2).sum()) + 1e-12
        out[h] = float(num / den)
    return out


def gradient_cosine_vs_horizon(reader: TrajectoryReader, layer: str, horizons: List[int]) -> Dict[int, float]:
    G = reader.layer_field(layer, "G")
    cad = reader.cadence_for(f"layer__{layer}__G")
    out = {}
    for h in horizons:
        hi = max(1, int(np.ceil(h / cad)))
        if hi >= len(G):
            out[h] = float("nan")
            continue
        vals = [_cos(G[t], G[t + hi]) for t in range(len(G) - hi)]
        out[h] = float(np.nanmean(vals))
    return out


def update_cosine_vs_horizon(reader: TrajectoryReader, layer: str, horizons: List[int]) -> Dict[int, float]:
    dW = reader.layer_field(layer, "DeltaW")
    cad = reader.cadence_for(f"layer__{layer}__DeltaW")
    out = {}
    for h in horizons:
        hi = max(1, int(np.ceil(h / cad)))
        if hi >= len(dW):
            out[h] = float("nan")
            continue
        out[h] = float(np.nanmean([_cos(dW[t], dW[t + hi]) for t in range(len(dW) - hi)]))
    return out


def subspace_overlap_vs_horizon(reader, layer, horizons, rank=8) -> Dict[int, float]:
    G = reader.layer_field(layer, "G")
    cad = reader.cadence_for(f"layer__{layer}__G")
    out = {}
    for h in horizons:
        hi = max(1, int(np.ceil(h / cad)))
        if hi >= len(G):
            out[h] = float("nan")
            continue
        vals = []
        for t in range(len(G) - hi):
            U1, _, _ = linalg.top_singular(torch.from_numpy(G[t]), min(rank, *G[t].shape))
            U2, _, _ = linalg.top_singular(torch.from_numpy(G[t + hi]), min(rank, *G[t + hi].shape))
            vals.append(linalg.subspace_overlap(U1, U2))
        out[h] = float(np.nanmean(vals))
    return out


def energy_capture_vs_horizon(reader, layer, horizons, ranks=(1, 2, 4, 8, 16)) -> Dict[int, Dict[int, float]]:
    """R_{t+h|t}: energy of future gradient captured by current subspace."""
    G = reader.layer_field(layer, "G")
    cad_g = reader.cadence_for(f"layer__{layer}__G")
    out: Dict[int, Dict[int, float]] = {}
    for r in ranks:
        out[r] = {}
        for h in horizons:
            hi = max(1, int(np.ceil(h / cad_g)))
            vals = []
            for t in range(len(G) - hi):
                rank = min(r, *G[t].shape)
                U, _, V = linalg.top_singular(torch.from_numpy(G[t]), rank)
                Gh = torch.from_numpy(G[t + hi])
                vals.append(linalg.energy_capture(U, Gh, V))
            out[r][h] = float(np.nanmean(vals)) if vals else float("nan")
    return out


def singular_value_trajectories(reader, layer) -> np.ndarray:
    return reader.layer_field(layer, "top_singular_values_W")


def effective_rank_trajectory(reader, layer) -> np.ndarray:
    W = reader.layer_field(layer, "W")
    return np.array([linalg.effective_rank(torch.from_numpy(w)) for w in W])


def phase_of(progress: float) -> str:
    for (lo, hi), name in zip(PHASES, PHASE_NAMES):
        if lo <= progress < hi:
            return name
    return PHASE_NAMES[-1]


def analytic_contraction_check(reader) -> Dict[str, float]:
    """Verify G_1 = W_2^T G_M and G_2 = G_M W_1^T for a 2-layer linear net (H4)."""
    names = reader.layer_names
    if len(names) != 2 or not reader.has_field("network__end_to_end_gradient"):
        return {}
    W1 = reader.layer_field(names[0], "W")
    W2 = reader.layer_field(names[1], "W")
    G1 = reader.layer_field(names[0], "G")
    G2 = reader.layer_field(names[1], "G")
    GM = reader.network_field("end_to_end_gradient")
    n = min(len(W1), len(GM))
    e1, e2 = [], []
    for t in range(n):
        pred_G1 = W2[t].T @ GM[t]
        pred_G2 = GM[t] @ W1[t].T
        e1.append(linalg.normalized_mse(torch.from_numpy(pred_G1), torch.from_numpy(G1[t])))
        e2.append(linalg.normalized_mse(torch.from_numpy(pred_G2), torch.from_numpy(G2[t])))
    return {"G1_contraction_nmse": float(np.mean(e1)), "G2_contraction_nmse": float(np.mean(e2))}


def balance_trajectory(reader) -> np.ndarray:
    s = reader.scalars()
    if "balance_error_mean" in s.columns:
        return s["balance_error_mean"].to_numpy()
    return np.array([])


def full_temporal_report(reader, horizons=(1, 2, 4, 8, 16, 32, 64, 100), rank=8) -> dict:
    horizons = list(horizons)
    rep = {"layers": {}, "analytic_contraction": analytic_contraction_check(reader)}
    for layer in reader.layer_names:
        rep["layers"][layer] = {
            "gradient_cosine_vs_horizon": gradient_cosine_vs_horizon(reader, layer, horizons),
            "update_cosine_vs_horizon": update_cosine_vs_horizon(reader, layer, horizons),
            "subspace_overlap_vs_horizon": subspace_overlap_vs_horizon(reader, layer, horizons, rank),
            "energy_capture_vs_horizon": energy_capture_vs_horizon(reader, layer, horizons),
            "effective_rank_final": float(effective_rank_trajectory(reader, layer)[-1]),
        }
    return rep
