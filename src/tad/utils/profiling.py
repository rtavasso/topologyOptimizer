"""Compute accounting and timing (Sections 14.8, 15.3).

We track FLOPs at the granularity the spec needs for amortization arguments:
forward/backward passes, optimizer steps, predictor inference, and extra
candidate evaluations. Counts are approximate (dense matmul FLOPs) but
consistent across compared methods, which is what the protocol requires.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Dict


def matmul_flops(m: int, k: int, n: int) -> int:
    """FLOPs for an (m,k) x (k,n) matmul (multiply-add counted as 2)."""
    return 2 * m * k * n


@dataclass
class ComputeLedger:
    """Accumulates FLOPs by category and wall-clock by named timers."""

    flops: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    seconds: Dict[str, float] = field(default_factory=dict)

    def add_flops(self, category: str, amount: float, count: int = 1) -> None:
        self.flops[category] = self.flops.get(category, 0.0) + float(amount)
        self.counts[category] = self.counts.get(category, 0) + count

    def add_seconds(self, name: str, amount: float) -> None:
        self.seconds[name] = self.seconds.get(name, 0.0) + amount

    @contextmanager
    def timer(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add_seconds(name, time.perf_counter() - t0)

    def total_flops(self) -> float:
        return sum(self.flops.values())

    def to_dict(self) -> dict:
        return asdict(self)


def amortized_cost(
    trajectory_flops: float,
    predictor_train_flops: float,
    marginal_flops_per_run: float,
    n_target_runs: int,
) -> float:
    """C_amortized = (C_traj + C_pred_train) / N + C_marginal  (Section 14.8)."""
    n = max(1, n_target_runs)
    return (trajectory_flops + predictor_train_flops) / n + marginal_flops_per_run


def break_even_runs(
    fixed_overhead_flops: float,
    marginal_savings_per_run: float,
) -> float:
    """Number of target runs at which fixed overhead is recovered."""
    if marginal_savings_per_run <= 0:
        return float("inf")
    return fixed_overhead_flops / marginal_savings_per_run
