"""Nonlinear MLP for the immediate falsification test (Sections 6 Phase 1B/2).

f(x) = W_2 phi(W_1 x) and deeper variants. Exposes activation masks so the
effective local map J_t(x) = W_2 D_t(x) W_1 can be logged (Section 6 Phase 1B).
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn

from .base import TADModel

_ACTS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


class NonlinearMLP(TADModel):
    def __init__(self, cfg, seed: Optional[int] = None):
        super().__init__()
        dims: List[int] = list(cfg["dimensions"])
        bias = bool(cfg.get("bias", False))
        act_name = cfg.get("activation", "relu")
        self.dims = dims
        self.act_name = act_name
        gen = torch.Generator().manual_seed(seed if seed is not None else 0)

        linears = []
        self.acts = nn.ModuleList()
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            lin = nn.Linear(a, b, bias=bias)
            with torch.no_grad():
                bound = math.sqrt(6.0 / (a + b))
                lin.weight.uniform_(-bound, bound, generator=gen)
                if lin.bias is not None:
                    lin.bias.zero_()
            linears.append((f"W{i+1}", lin))
            # activation after every layer except the last
            self.acts.append(_ACTS[act_name]() if i < len(dims) - 2 else nn.Identity())
        self.layers = nn.ModuleList([lin for _, lin in linears])
        self.register_linears(linears)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, (lin, act) in enumerate(zip(self.layers, self.acts)):
            z = lin(h)
            h = act(z)
            if self._capture_enabled and not isinstance(act, nn.Identity):
                name = self.layer_names[i]
                # diagonal of the local Jacobian d act / d z (activation gating)
                with torch.no_grad():
                    if self.act_name == "relu":
                        self.captures[name].act_mask = (z > 0).float().mean(0)
                    else:
                        zc = z.detach().clone().requires_grad_(True)
                        a = self.acts[i](zc)
                        g, = torch.autograd.grad(a.sum(), zc)
                        self.captures[name].act_mask = g.mean(0)
        return h

    def effective_local_map(self, x: torch.Tensor) -> torch.Tensor:
        """Probe-averaged effective Jacobian J = W_L D ... D W_1 (Section 6.1B)."""
        was = self._capture_enabled
        self.enable_capture(True)
        _ = self.forward(x)
        J = None
        for i, (lin, act) in enumerate(zip(self.layers, self.acts)):
            W = lin.weight
            J = W if J is None else W @ J
            name = self.layer_names[i]
            mask = self.captures[name].act_mask
            if mask is not None and not isinstance(act, nn.Identity):
                J = mask.unsqueeze(1) * J
        self.enable_capture(was)
        return J
