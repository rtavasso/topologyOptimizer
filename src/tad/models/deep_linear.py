"""Deep linear student network (Sections 4.1, 6 Phase 1A).

f(x) = W_L ... W_1 x. No biases by default. Supports the spec's initialization
modes including balanced / unbalanced factorizations.
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .base import TADModel


class DeepLinear(TADModel):
    def __init__(self, cfg, seed: Optional[int] = None):
        super().__init__()
        dims: List[int] = list(cfg["dimensions"])
        assert len(dims) >= 2, "need at least input and output dims"
        bias = bool(cfg.get("bias", False))
        self.dims = dims
        gen = torch.Generator().manual_seed(seed if seed is not None else 0)

        linears = []
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            lin = nn.Linear(a, b, bias=bias)
            linears.append((f"W{i+1}", lin))
        self.layers = nn.ModuleList([lin for _, lin in linears])
        self.register_linears(linears)
        self._init_weights(cfg.get("initialization", "xavier"), gen)

    def _init_weights(self, mode: str, gen: torch.Generator):
        for name in self.layer_names:
            W = self._linears[name].weight
            out_d, in_d = W.shape
            with torch.no_grad():
                if mode == "gaussian":
                    W.normal_(0.0, 1.0 / math.sqrt(in_d), generator=gen)
                elif mode == "xavier":
                    bound = math.sqrt(6.0 / (in_d + out_d))
                    W.uniform_(-bound, bound, generator=gen)
                elif mode == "orthogonal":
                    nn.init.orthogonal_(W)
                elif mode == "identity_plus_noise":
                    eye = torch.zeros_like(W)
                    m = min(out_d, in_d)
                    eye[:m, :m] = torch.eye(m)
                    W.copy_(eye + 0.01 * torch.randn(W.shape, generator=gen))
                elif mode in ("balanced_svd", "unbalanced"):
                    self._factorized_init(mode, gen)
                    return
                else:
                    raise ValueError(f"unknown initialization: {mode}")
                if self._linears[name].bias is not None:
                    self._linears[name].bias.zero_()

    def _factorized_init(self, mode: str, gen: torch.Generator):
        """Balanced/unbalanced factorization of a random end-to-end map."""
        dims = self.dims
        d_out, d_in = dims[-1], dims[0]
        r = min(dims)
        n_layers = len(self.layer_names)
        U, _ = torch.linalg.qr(torch.randn(d_out, r, generator=gen), mode="reduced")
        V, _ = torch.linalg.qr(torch.randn(d_in, r, generator=gen), mode="reduced")
        S = torch.exp(torch.randn(r, generator=gen) * 0.25) / math.sqrt(d_in)
        if mode == "balanced_svd":
            s_each = S.clamp_min(1e-8) ** (1.0 / n_layers)
        else:  # unbalanced: concentrate scale in first layer
            s_each = S.clamp_min(1e-8)
        with torch.no_grad():
            for i, name in enumerate(self.layer_names):
                W = self._linears[name].weight
                out_d, in_d = W.shape
                block = torch.zeros(out_d, in_d)
                if i == 0:
                    diagv = s_each if mode == "balanced_svd" else S
                    block[:r, :] = torch.diag(diagv) @ V.T
                elif i == n_layers - 1:
                    diagv = s_each if mode == "balanced_svd" else torch.ones_like(S)
                    block[:, :r] = U @ torch.diag(diagv)
                else:
                    diagv = s_each if mode == "balanced_svd" else torch.ones_like(S)
                    block[:r, :r] = torch.diag(diagv)
                W.copy_(block)
                if self._linears[name].bias is not None:
                    self._linears[name].bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for lin in self.layers:
            h = lin(h)
        return h
