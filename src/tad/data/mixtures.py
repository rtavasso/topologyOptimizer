"""Batch heterogeneity generator (Section 5.4).

A mixture batch draws each sample from one of several components (common, rare,
corrupted, ...). Each component has its own covariance factor, teacher, and
noise. Component assignments are returned so the logger can record per-batch
surprise composition (needed by the innovation-decomposition analysis, 13.8).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .covariance import build_covariance_factor
from .teachers import build_teacher


@dataclass
class MixtureComponent:
    name: str
    probability: float
    factor: np.ndarray
    teacher_matrix: np.ndarray
    noise_std: float


class MixtureSampler:
    def __init__(self, components: List[MixtureComponent], in_dim: int, out_dim: int):
        self.components = components
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.probs = np.array([c.probability for c in components], dtype=np.float64)
        self.probs = self.probs / self.probs.sum()

    def sample(self, batch_size: int, rng: np.random.Generator):
        comp_idx = rng.choice(len(self.components), size=batch_size, p=self.probs)
        x = np.empty((batch_size, self.in_dim), dtype=np.float32)
        y = np.empty((batch_size, self.out_dim), dtype=np.float32)
        for k, comp in enumerate(self.components):
            sel = np.where(comp_idx == k)[0]
            if sel.size == 0:
                continue
            z = rng.standard_normal((sel.size, self.in_dim))
            xc = z @ comp.factor.T
            yc = xc @ comp.teacher_matrix.T + comp.noise_std * rng.standard_normal((sel.size, self.out_dim))
            x[sel] = xc.astype(np.float32)
            y[sel] = yc.astype(np.float32)
        return x, y, comp_idx


def build_mixture(cfg: dict, in_dim: int, out_dim: int, rng: np.random.Generator) -> MixtureSampler:
    comps = []
    for c in cfg["components"]:
        factor = build_covariance_factor(
            {"type": c.get("covariance", "identity")} if isinstance(c.get("covariance"), str)
            else c.get("covariance", {"type": "identity"}),
            in_dim, rng,
        )
        teacher = build_teacher(c.get("teacher", {"type": "random_gaussian"}), in_dim, out_dim, rng)
        comps.append(MixtureComponent(
            name=c.get("name", "comp"),
            probability=float(c.get("probability", 1.0)),
            factor=factor,
            teacher_matrix=teacher.matrix(),
            noise_std=float(c.get("noise_std", 0.01)),
        ))
    return MixtureSampler(comps, in_dim, out_dim)
