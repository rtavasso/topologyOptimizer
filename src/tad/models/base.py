"""Base model with per-layer activation / backprop-error capture.

Every TAD model exposes an ordered list of linear maps ``W_ell`` (the topology
nodes) and, after a forward+backward pass, the inputs to each map and the
gradient of the loss w.r.t. each map's pre-activation output. These feed the
per-layer logging fields in Section 8.3 (activation/error covariances, effective
local maps for nonlinear nets, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class LayerCapture:
    """Captured tensors for one linear map during the last forward/backward."""

    pre_input: Optional[torch.Tensor] = None       # input to the map (B, in)
    pre_output: Optional[torch.Tensor] = None      # output of the map (B, out)
    out_grad: Optional[torch.Tensor] = None        # dL/d(pre_output) (B, out)
    act_mask: Optional[torch.Tensor] = None        # activation gating diag, nonlinear only


class TADModel(nn.Module):
    """Common interface: an ordered list of named linear maps + capture hooks."""

    def __init__(self):
        super().__init__()
        self.layer_names: List[str] = []
        self._linears: Dict[str, nn.Linear] = {}
        self.captures: Dict[str, LayerCapture] = {}
        self._hooks = []
        self._capture_enabled = False

    # -- topology accessors --------------------------------------------------
    def register_linears(self, named_linears: List[tuple]):
        for name, lin in named_linears:
            self.layer_names.append(name)
            self._linears[name] = lin

    def linear(self, name: str) -> nn.Linear:
        return self._linears[name]

    def weight(self, name: str) -> torch.Tensor:
        return self._linears[name].weight

    def weights(self) -> Dict[str, torch.Tensor]:
        return {n: self._linears[n].weight for n in self.layer_names}

    def end_to_end_map(self) -> torch.Tensor:
        """Product W_L ... W_1 for purely sequential linear stacks.

        For nonlinear models this is the product of the *linear* maps only and
        should be interpreted alongside the effective-Jacobian fields.
        """
        M = None
        for name in self.layer_names:
            W = self._linears[name].weight
            M = W if M is None else W @ M
        return M

    # -- capture machinery ---------------------------------------------------
    def enable_capture(self, enabled: bool = True):
        self._capture_enabled = enabled
        if enabled and not self._hooks:
            self._install_hooks()

    def _install_hooks(self):
        for name, lin in self._linears.items():
            self.captures[name] = LayerCapture()

            def fwd_hook(mod, inp, out, _name=name):
                if not self._capture_enabled:
                    return
                cap = self.captures[_name]
                cap.pre_input = inp[0].detach()
                cap.pre_output = out.detach()
                if out.requires_grad:
                    def grad_hook(g, _n=_name):
                        if self._capture_enabled:
                            self.captures[_n].out_grad = g.detach()
                    out.register_hook(grad_hook)

            self._hooks.append(lin.register_forward_hook(fwd_hook))
