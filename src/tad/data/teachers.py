"""Teacher matrix families (Section 5.3).

A teacher maps inputs to (noiseless) targets: ``y = M_* x``. Builders return a
numpy matrix ``M_*`` of shape (output_dim, input_dim). Drifting / regime
teachers expose a ``__call__(progress, regime)`` via :class:`Teacher`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _prescribed_spectrum(out_dim: int, in_dim: int, rank: int, spectrum: str,
                         cond: float, rng: np.random.Generator) -> np.ndarray:
    r = min(rank, out_dim, in_dim)
    if spectrum == "geometric":
        s = np.geomspace(1.0, 1.0 / cond, num=r)
    elif spectrum == "linear":
        s = np.linspace(1.0, 1.0 / cond, num=r)
    elif spectrum == "flat":
        s = np.ones(r)
    else:
        raise ValueError(f"unknown spectrum: {spectrum}")
    U = np.linalg.qr(rng.standard_normal((out_dim, r)))[0]
    V = np.linalg.qr(rng.standard_normal((in_dim, r)))[0]
    return (U * s[None, :]) @ V.T


def build_teacher_matrix(cfg: dict, in_dim: int, out_dim: int, rng: np.random.Generator) -> np.ndarray:
    ttype = cfg.get("type", "random_gaussian")

    if ttype == "random_gaussian":
        return rng.standard_normal((out_dim, in_dim)) / np.sqrt(in_dim)

    if ttype in ("orthogonal", "semi_orthogonal"):
        A = rng.standard_normal((out_dim, in_dim))
        U, _, Vh = np.linalg.svd(A, full_matrices=False)
        return U @ Vh

    if ttype == "prescribed_rank":
        rank = int(cfg.get("rank", min(in_dim, out_dim)))
        return _prescribed_spectrum(out_dim, in_dim, rank, "flat", 1.0, rng)

    if ttype == "prescribed_spectrum":
        return _prescribed_spectrum(
            out_dim, in_dim,
            int(cfg.get("rank", min(in_dim, out_dim))),
            cfg.get("spectrum", "geometric"),
            float(cfg.get("condition_number", 10.0)),
            rng,
        )

    if ttype == "block_diagonal":
        nb = int(cfg.get("num_blocks", 4))
        M = np.zeros((out_dim, in_dim))
        ob = np.array_split(np.arange(out_dim), nb)
        ib = np.array_split(np.arange(in_dim), nb)
        for o, i in zip(ob, ib):
            M[np.ix_(o, i)] = rng.standard_normal((len(o), len(i))) / np.sqrt(max(1, len(i)))
        return M

    if ttype == "sparse":
        density = float(cfg.get("density", 0.1))
        M = rng.standard_normal((out_dim, in_dim)) / np.sqrt(in_dim)
        mask = rng.random((out_dim, in_dim)) < density
        return M * mask

    if ttype == "low_rank_experts":
        n_exp = int(cfg.get("num_experts", 3))
        er = int(cfg.get("expert_rank", 2))
        M = np.zeros((out_dim, in_dim))
        for _ in range(n_exp):
            A = rng.standard_normal((out_dim, er)) / np.sqrt(er)
            B = rng.standard_normal((in_dim, er)) / np.sqrt(in_dim)
            M += A @ B.T
        return M

    if ttype == "compositional":
        depth = int(cfg.get("depth", 3))
        dims = cfg.get("hidden_dims", [max(in_dim, out_dim)] * (depth - 1))
        dims = [in_dim] + list(dims) + [out_dim]
        M = np.eye(in_dim)
        for a, b in zip(dims[:-1], dims[1:]):
            T = rng.standard_normal((b, a)) / np.sqrt(a)
            M = T @ M
        return M

    if ttype == "random_labels":
        # Used by the corrupted batch component: returns a fresh random map.
        return rng.standard_normal((out_dim, in_dim)) / np.sqrt(in_dim)

    raise ValueError(f"unknown teacher type: {ttype}")


@dataclass
class Teacher:
    """A (possibly drifting / regime-conditioned) teacher (5.3.9, 5.3.10)."""

    M0: np.ndarray
    deltas: Optional[np.ndarray] = None  # (num_regimes, out, in) for regime teacher
    drift_std: float = 0.0
    drift_steps: int = 0
    cfg: dict = None

    def matrix(self, progress: float = 0.0, regime: int = 0, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        M = self.M0
        if self.deltas is not None and regime > 0 and regime <= len(self.deltas):
            M = M + self.deltas[regime - 1]
        if self.drift_std > 0 and self.drift_steps > 0:
            # deterministic slow drift parameterised by progress
            step = int(progress * self.drift_steps)
            g = np.random.default_rng(12345)
            acc = M.copy()
            for _ in range(step):
                acc = acc + self.drift_std * g.standard_normal(M.shape)
                acc = acc / (np.linalg.norm(acc) / (np.linalg.norm(M) + 1e-9) + 1e-9)
            M = acc
        return M


def build_teacher(cfg: dict, in_dim: int, out_dim: int, rng: np.random.Generator) -> Teacher:
    ttype = cfg.get("type", "random_gaussian")
    M0 = build_teacher_matrix(cfg, in_dim, out_dim, rng)
    deltas = None
    if ttype == "regime_conditioned" or "regime_deltas" in cfg:
        nreg = int(cfg.get("num_regimes", 2))
        scale = float(cfg.get("regime_scale", 0.3))
        deltas = np.stack([scale * rng.standard_normal((out_dim, in_dim)) for _ in range(nreg)])
    return Teacher(
        M0=M0,
        deltas=deltas,
        drift_std=float(cfg.get("drift_std", 0.0)),
        drift_steps=int(cfg.get("drift_steps", 0)),
        cfg=dict(cfg),
    )
