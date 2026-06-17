"""Candidate-update isolation and held-out selection (Sections 14.1, 20.4).

A candidate update is applied temporarily, evaluated on a held-out selection
microbatch, then fully rolled back: parameters, optimizer state, RNG state, and
gradients are all restored (tested in test_candidate_updates.py).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Dict, List

import torch

from ..utils.seeds import RngSnapshot


@dataclass
class CandidateContext:
    """Snapshots the full mutable training state for exact restoration."""

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    _param_backup: Dict[str, torch.Tensor] = None
    _opt_backup: dict = None
    _grad_backup: Dict[str, torch.Tensor] = None
    _rng: RngSnapshot = None

    def __enter__(self):
        self._param_backup = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self._opt_backup = copy.deepcopy(self.optimizer.state_dict())
        self._grad_backup = {
            n: (p.grad.detach().clone() if p.grad is not None else None)
            for n, p in self.model.named_parameters()
        }
        self._rng = RngSnapshot.capture()
        return self

    def restore(self):
        with torch.no_grad():
            sd = self.model.state_dict()
            for k, v in self._param_backup.items():
                sd[k].copy_(v)
        self.optimizer.load_state_dict(self._opt_backup)
        for n, p in self.model.named_parameters():
            g = self._grad_backup.get(n)
            p.grad = None if g is None else g.clone()
        self._rng.restore()

    def __exit__(self, *exc):
        self.restore()
        return False


def apply_delta(model: torch.nn.Module, deltas: Dict[str, torch.Tensor]) -> None:
    """Add per-parameter deltas in place (keys = named_parameters keys)."""
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in deltas:
                p.add_(deltas[n])


def evaluate_candidates(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    candidates: List[Dict[str, torch.Tensor]],
    loss_fn: Callable[[], float],
) -> dict:
    """Evaluate held-out selection loss for each candidate, restoring between.

    Returns winner index, per-candidate losses, and the held-out improvement of
    the winner over candidate 0 (the tuned-baseline update, by convention).
    """
    losses = []
    for delta in candidates:
        with CandidateContext(model, optimizer):
            apply_delta(model, delta)
            losses.append(float(loss_fn()))
    losses_t = torch.tensor(losses)
    k_star = int(losses_t.argmin().item())
    return {
        "losses": losses,
        "winner": k_star,
        "baseline_loss": losses[0],
        "winner_loss": losses[k_star],
        "improvement_over_baseline": losses[0] - losses[k_star],
    }
