"""Prefix/suffix products and end-to-end map (Section 8.4)."""
from __future__ import annotations

from typing import Dict, List

import torch


def prefix_products(weights: List[torch.Tensor]) -> List[torch.Tensor]:
    """P_l = W_l W_{l-1} ... W_1 for l = 1..L."""
    out = []
    P = None
    for W in weights:
        P = W if P is None else W @ P
        out.append(P)
    return out


def suffix_products(weights: List[torch.Tensor]) -> List[torch.Tensor]:
    """S_l = W_L W_{L-1} ... W_l for l = 1..L."""
    out = [None] * len(weights)
    S = None
    for i in range(len(weights) - 1, -1, -1):
        W = weights[i]
        S = W if S is None else S @ W
        out[i] = S
    return out


def end_to_end_map(weights: List[torch.Tensor]) -> torch.Tensor:
    return prefix_products(weights)[-1]


def gram_pair(W_curr: torch.Tensor, W_next: torch.Tensor):
    """Return (W^T W, W_next W_next^T) used in balance and R5 features."""
    return W_curr.transpose(-2, -1) @ W_curr, W_next @ W_next.transpose(-2, -1)


def product_update_decomposition(W2, W1, dW2, dW1):
    """Exact M update terms (Section 20.2):

    M_{t+1} - M_t = dW2 W1 + W2 dW1 + dW2 dW1.
    """
    return dW2 @ W1, W2 @ dW1, dW2 @ dW1
