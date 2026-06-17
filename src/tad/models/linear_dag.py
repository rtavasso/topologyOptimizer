"""Residual / branched linear networks (Section 6 Phase 1A).

Residual linear: h_{l+1} = h_l + W_l h_l. Implemented as a TADModel so the same
logging/topology machinery applies; the topology graph records the residual edge.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .base import TADModel


class ResidualLinear(TADModel):
    def __init__(self, cfg, seed: Optional[int] = None):
        super().__init__()
        dims = list(cfg["dimensions"])
        assert all(d == dims[0] for d in dims), "residual linear requires constant width"
        bias = bool(cfg.get("bias", False))
        self.dims = dims
        gen = torch.Generator().manual_seed(seed if seed is not None else 0)
        linears = []
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            lin = nn.Linear(a, b, bias=bias)
            with torch.no_grad():
                bound = math.sqrt(6.0 / (a + b))
                lin.weight.uniform_(-bound, bound, generator=gen)
            linears.append((f"W{i+1}", lin))
        self.layers = nn.ModuleList([lin for _, lin in linears])
        self.register_linears(linears)
        self.residual = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for lin in self.layers:
            h = h + lin(h)
        return h
