"""Curriculum and distribution-shift schedules (Section 5.5).

A schedule maps normalized training progress ``p in [0, 1]`` to a regime id and
optional difficulty / noise multipliers. Every step's regime is logged so the
predictor can be conditioned (and evaluated) on data regime.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RegimeState:
    regime_id: int
    noise_mult: float = 1.0
    difficulty: float = 1.0


def build_schedule(cfg: dict):
    cfg = cfg or {"type": "stationary"}
    stype = cfg.get("type", "stationary")

    if stype == "stationary":
        return lambda p: RegimeState(0)

    if stype == "increasing_noise":
        lo, hi = float(cfg.get("min_mult", 1.0)), float(cfg.get("max_mult", 5.0))
        return lambda p: RegimeState(0, noise_mult=lo + (hi - lo) * p)

    if stype == "decreasing_noise":
        lo, hi = float(cfg.get("min_mult", 1.0)), float(cfg.get("max_mult", 5.0))
        return lambda p: RegimeState(0, noise_mult=hi - (hi - lo) * p)

    if stype == "easy_to_hard":
        return lambda p: RegimeState(0, difficulty=0.2 + 0.8 * p)

    if stype == "hard_to_easy":
        return lambda p: RegimeState(0, difficulty=1.0 - 0.8 * p)

    if stype == "abrupt_switch":
        at = float(cfg.get("switch_at", 0.5))
        return lambda p: RegimeState(0 if p < at else 1)

    if stype == "periodic_alternation":
        period = float(cfg.get("period", 0.1))
        return lambda p: RegimeState(int((p / period) % 2))

    if stype == "gradual_rotation":
        # regime id stays 0; the covariance/teacher use progress directly.
        return lambda p: RegimeState(0, difficulty=p)

    raise ValueError(f"unknown schedule type: {stype}")
