"""Balancedness / conserved-quantity invariants (Sections 8.4, 13.11, 20.6).

For adjacent linear layers the gradient-flow conserved quantity is
B_l = W_{l+1}^T W_{l+1} - W_l W_l^T. Under continuous gradient flow with no
regularization dB_l/dt = 0. Under finite-step SGD / AdamW it drifts; we measure
preservation relative to initialization rather than assuming restoration
(interpretation rule 11).
"""
from __future__ import annotations

from typing import List

import torch


def balance_error(W_curr: torch.Tensor, W_next: torch.Tensor) -> torch.Tensor:
    """||W_{l+1}^T W_{l+1} - W_l W_l^T||_F (requires matching inner dim)."""
    A = W_next.transpose(-2, -1) @ W_next
    B = W_curr @ W_curr.transpose(-2, -1)
    if A.shape != B.shape:
        # dimensions only line up for square/compatible adjacent layers; return nan
        return torch.tensor(float("nan"))
    return torch.linalg.norm(A - B)


def all_balance_errors(weights: List[torch.Tensor]) -> List[float]:
    errs = []
    for Wc, Wn in zip(weights[:-1], weights[1:]):
        errs.append(float(balance_error(Wc, Wn).item()))
    return errs


def balance_invariant_matrix(W_curr: torch.Tensor, W_next: torch.Tensor) -> torch.Tensor:
    """The conserved matrix B_l (not its norm), for drift tracking."""
    return W_next.transpose(-2, -1) @ W_next - W_curr @ W_curr.transpose(-2, -1)
