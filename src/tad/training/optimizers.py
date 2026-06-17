"""Optimizer factory + serializable snapshots (Section 7.1).

Exposes SGD, momentum, Nesterov, Adam, AdamW, and a minimal Muon for matrix
parameters. Every optimizer exposes per-parameter state (momentum / second
moment) so the logger can record optimizer_momentum_t and
optimizer_second_moment_t (Section 8.3).
"""
from __future__ import annotations

from typing import Dict, List

import torch


class Muon(torch.optim.Optimizer):
    """Minimal Muon: momentum + Newton-Schulz orthogonalization of the update.

    Applies only to 2D parameters; others fall back to momentum SGD. This is the
    practical-where-possible variant referenced in Section 7.1.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps,
                                      weight_decay=weight_decay))

    @staticmethod
    def _newton_schulz(G, steps):
        a, b, c = 3.4445, -4.7750, 2.0315
        X = G.bfloat16() if G.is_cuda else G.float()
        X = X / (X.norm() + 1e-7)
        transpose = X.shape[0] > X.shape[1]
        if transpose:
            X = X.T
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        if transpose:
            X = X.T
        return X.to(G.dtype)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(g)
                if p.ndim == 2:
                    update = self._newton_schulz(buf, group["ns_steps"])
                    scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                    update = update * scale
                else:
                    update = buf
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])
        return loss


def build_optimizer(cfg, params: List[torch.nn.Parameter]) -> torch.optim.Optimizer:
    otype = cfg.get("type", "adamw")
    lr = float(cfg.get("learning_rate", 1e-3))
    wd = float(cfg.get("weight_decay", 0.0))
    if otype == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.0, weight_decay=wd)
    if otype == "momentum":
        return torch.optim.SGD(params, lr=lr, momentum=float(cfg.get("momentum", 0.9)),
                               weight_decay=wd)
    if otype == "nesterov":
        return torch.optim.SGD(params, lr=lr, momentum=float(cfg.get("momentum", 0.9)),
                               nesterov=True, weight_decay=wd)
    if otype == "adam":
        return torch.optim.Adam(params, lr=lr, betas=tuple(cfg.get("betas", [0.9, 0.999])),
                                weight_decay=wd)
    if otype == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=tuple(cfg.get("betas", [0.9, 0.999])),
                                 weight_decay=wd)
    if otype == "muon":
        return Muon(params, lr=lr, momentum=float(cfg.get("momentum", 0.95)),
                    weight_decay=wd)
    raise ValueError(f"unknown optimizer type: {otype}")


def optimizer_state_snapshot(optimizer, param_to_name: Dict[int, str]) -> Dict[str, Dict[str, torch.Tensor]]:
    """Map each parameter's optimizer state to {name: {momentum, second_moment}}."""
    snap: Dict[str, Dict[str, torch.Tensor]] = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            name = param_to_name.get(id(p))
            if name is None:
                continue
            st = optimizer.state.get(p, {})
            entry = {}
            if "exp_avg" in st:
                entry["momentum"] = st["exp_avg"].detach().clone()
            if "momentum_buffer" in st:
                entry["momentum"] = st["momentum_buffer"].detach().clone()
            if "exp_avg_sq" in st:
                entry["second_moment"] = st["exp_avg_sq"].detach().clone()
            snap[name] = entry
    return snap
