"""Spectral summaries for logging (Section 8.3 SVD fields)."""
from __future__ import annotations

from typing import Dict

import torch

from ..utils import linalg


def spectral_fields(name: str, W, G, dW, rank: int) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    Uw, Sw, Vw = linalg.top_singular(W, rank)
    out["top_singular_values_W"] = Sw
    if G is not None:
        Ug, Sg, Vg = linalg.top_singular(G, rank)
        out["top_singular_values_G"] = Sg
        out["top_left_singular_vectors_G"] = Ug
        out["top_right_singular_vectors_G"] = Vg
    if dW is not None:
        Ud, Sd, Vd = linalg.top_singular(dW, rank)
        out["top_singular_values_DeltaW"] = Sd
        out["top_left_singular_vectors_DeltaW"] = Ud
        out["top_right_singular_vectors_DeltaW"] = Vd
    return out
