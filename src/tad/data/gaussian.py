"""Gaussian matrix-regression data stream (Sections 5.1, 5.6).

Batches are generated deterministically from ``(seed, step)`` so the exact same
sequence can be replayed. Supports static / time-varying covariance, fixed /
drifting / regime teachers, curriculum schedules, and mixture batches.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch

from ..utils.seeds import derive_seed
from .base import Batch
from .covariance import build_covariance_factor, build_time_varying_factor
from .mixtures import build_mixture
from .schedules import build_schedule
from .teachers import build_teacher


class GaussianMatrixRegression:
    def __init__(self, cfg, seed: int, total_steps: int, device: str = "cpu"):
        self.cfg = cfg
        self.seed = seed
        self.total_steps = max(1, total_steps)
        self.device = device
        self.input_dim = int(cfg["input_dim"])
        self.output_dim = int(cfg["output_dim"])
        self.batch_size = int(cfg["batch_size"])
        self.noise_std = float(cfg.get("noise_std", 0.01))

        # A dedicated RNG for fixed structures (covariance/teacher/probes) keyed
        # only by seed, so structures are identical across replays.
        struct_rng = np.random.default_rng(derive_seed(seed, "structures"))

        cov_cfg = dict(cfg.get("input_covariance", {"type": "identity"}))
        if cov_cfg.get("type") == "time_varying":
            self._cov_factor_fn: Callable[[float], np.ndarray] = build_time_varying_factor(
                cov_cfg, self.input_dim, struct_rng)
            self._static_factor = None
        else:
            self._static_factor = build_covariance_factor(cov_cfg, self.input_dim, struct_rng)
            self._cov_factor_fn = None

        self.teacher = build_teacher(cfg.get("teacher", {"type": "random_gaussian"}),
                                     self.input_dim, self.output_dim, struct_rng)
        self.schedule = build_schedule(dict(cfg.get("schedule", {"type": "stationary"})))

        mix_cfg = cfg.get("batch_mixture")
        self.mixture = build_mixture(dict(mix_cfg), self.input_dim, self.output_dim, struct_rng) if mix_cfg else None

        # Validation set is fixed for the run.
        self._val = self._make_validation(int(cfg.get("validation_size", 4096)))

    # -- structure accessors -------------------------------------------------
    def teacher_matrix(self, progress: float = 0.0, regime: int = 0) -> np.ndarray:
        return self.teacher.matrix(progress, regime)

    def _factor(self, progress: float) -> np.ndarray:
        return self._static_factor if self._cov_factor_fn is None else self._cov_factor_fn(progress)

    # -- batch generation ----------------------------------------------------
    def batch(self, step: int) -> Batch:
        progress = step / self.total_steps
        regime = self.schedule(progress)
        rng = np.random.default_rng(derive_seed(self.seed, "batch", step))

        if self.mixture is not None:
            x, y, _ = self.mixture.sample(self.batch_size, rng)
        else:
            factor = self._factor(progress)
            z = rng.standard_normal((self.batch_size, self.input_dim))
            x = (z @ factor.T).astype(np.float32)
            M = self.teacher_matrix(progress, regime.regime_id)
            noise = (self.noise_std * regime.noise_mult) * rng.standard_normal((self.batch_size, self.output_dim))
            y = (x @ M.T + noise).astype(np.float32)

        return Batch(
            x=torch.from_numpy(x).to(self.device),
            y=torch.from_numpy(y).to(self.device),
            regime_id=regime.regime_id,
            step=step,
        )

    def _make_validation(self, n: int) -> Batch:
        rng = np.random.default_rng(derive_seed(self.seed, "validation"))
        factor = self._factor(0.0)
        z = rng.standard_normal((n, self.input_dim))
        x = (z @ factor.T).astype(np.float32)
        M = self.teacher_matrix(0.0, 0)
        y = (x @ M.T).astype(np.float32)  # noiseless validation targets
        return Batch(
            x=torch.from_numpy(x).to(self.device),
            y=torch.from_numpy(y).to(self.device),
            regime_id=0,
            step=-1,
        )

    def validation_set(self) -> Batch:
        return self._val


def build_data_stream(cfg, seed: int, total_steps: int, device: str = "cpu"):
    dtype = cfg.get("type", "gaussian_matrix_regression")
    if dtype == "gaussian_matrix_regression":
        return GaussianMatrixRegression(cfg, seed, total_steps, device)
    raise ValueError(f"unknown data type: {dtype}")
