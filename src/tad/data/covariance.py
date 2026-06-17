"""Input covariance families (Section 5.2).

Each builder returns a Cholesky-like factor ``L`` with ``Sigma = L L^T`` so a
sample is ``x = L z`` for ``z ~ N(0, I)``. Time-varying covariance is supported
by returning a callable factor parameterised by normalized training progress.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

CovFactor = np.ndarray  # (d, d) lower factor such that Sigma = L L^T


def _factor_from_spectrum(eigvals: np.ndarray, rng: np.random.Generator, rotate: bool) -> CovFactor:
    d = eigvals.shape[0]
    if rotate:
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    else:
        Q = np.eye(d)
    # Sigma = Q diag(eig) Q^T => factor L = Q diag(sqrt(eig))
    return (Q * np.sqrt(np.maximum(eigvals, 0.0))[None, :]) @ np.eye(d)


def build_covariance_factor(cfg: dict, dim: int, rng: np.random.Generator) -> CovFactor:
    """Static covariance factor for the supported families."""
    ctype = cfg.get("type", "identity")
    if ctype == "identity":
        return np.eye(dim)

    if ctype == "power_law":
        alpha = float(cfg.get("alpha", 1.0))
        idx = np.arange(1, dim + 1, dtype=np.float64)
        eig = idx ** (-alpha)
        eig = eig / eig.mean()  # keep average variance ~1
        return _factor_from_spectrum(eig, rng, rotate=bool(cfg.get("rotate", False)))

    if ctype == "spiked":
        r_s = int(cfg.get("num_spikes", 4))
        gamma = float(cfg.get("spike_strength", 5.0))
        U = np.linalg.qr(rng.standard_normal((dim, r_s)))[0]
        Sigma = np.eye(dim) + gamma * (U @ U.T)
        evals, evecs = np.linalg.eigh(Sigma)
        return (evecs * np.sqrt(np.maximum(evals, 0.0))[None, :])

    if ctype == "random_rotated_diagonal":
        eig = rng.uniform(float(cfg.get("min_eig", 0.1)), float(cfg.get("max_eig", 3.0)), size=dim)
        return _factor_from_spectrum(eig, rng, rotate=True)

    if ctype == "low_rank_manifold":
        r = int(cfg.get("manifold_rank", 8))
        noise = float(cfg.get("ambient_noise", 0.1))
        A = rng.standard_normal((dim, r)) / np.sqrt(r)
        Sigma = A @ A.T + (noise ** 2) * np.eye(dim)
        evals, evecs = np.linalg.eigh(Sigma)
        return (evecs * np.sqrt(np.maximum(evals, 0.0))[None, :])

    raise ValueError(f"unknown covariance type: {ctype}")


def build_time_varying_factor(cfg: dict, dim: int, rng: np.random.Generator) -> Callable[[float], CovFactor]:
    """Return ``factor(progress)`` for rotating / switching covariance (5.2.7).

    ``mode``:
      - ``rotate``: smoothly rotate the eigenbasis between two random frames.
      - ``switch``: abrupt switch at ``switch_at`` progress.
    """
    base_cfg = dict(cfg.get("base", {"type": "identity"}))
    mode = cfg.get("mode", "rotate")
    L0 = build_covariance_factor(base_cfg, dim, rng)

    if mode == "switch":
        alt_cfg = dict(cfg.get("alt", {"type": "random_rotated_diagonal"}))
        L1 = build_covariance_factor(alt_cfg, dim, rng)
        switch_at = float(cfg.get("switch_at", 0.5))
        return lambda p: L0 if p < switch_at else L1

    # smooth rotation: interpolate an orthogonal generator
    G = rng.standard_normal((dim, dim))
    G = G - G.T  # skew-symmetric generator of a rotation
    G = G / (np.linalg.norm(G) + 1e-9)
    angle = float(cfg.get("max_angle", 1.0))

    def factor(p: float) -> CovFactor:
        from scipy.linalg import expm

        R = expm(angle * p * G)
        return R @ L0

    return factor
