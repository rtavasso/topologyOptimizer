"""Exact replay (Sections 5.6, 20.3).

Re-executes the deterministic training process from (config, seed) and returns
the batch tensors, losses, gradients, and parameters per step so two replays can
be asserted bit-for-bit identical on CPU (documented tolerance on GPU).
"""
from __future__ import annotations

from typing import List

import torch

from ..config import Config
from ..data import build_data_stream
from ..models import build_model
from ..utils.seeds import set_global_seed
from .optimizers import build_optimizer
from .trainer import mse_loss


def replay_run(cfg: Config, seed: int, n_steps: int, device: str = "cpu") -> dict:
    set_global_seed(seed)
    total_steps = int(cfg.data.get("steps", n_steps))
    stream = build_data_stream(cfg.data, seed, total_steps, device=device)
    model = build_model(cfg.model, seed=seed).to(device)
    optimizer = build_optimizer(cfg.optimizer, list(model.parameters()))

    batch_sigs: List[float] = []
    losses: List[float] = []
    grad_norms: List[float] = []
    param_sigs: List[float] = []

    for step in range(n_steps):
        batch = stream.batch(step)
        batch_sigs.append(float(batch.x.sum().item()) + float(batch.y.sum().item()))
        optimizer.zero_grad(set_to_none=True)
        loss = mse_loss(model(batch.x), batch.y)
        loss.backward()
        losses.append(float(loss.item()))
        gn = sum(float((p.grad ** 2).sum().item()) for p in model.parameters()) ** 0.5
        grad_norms.append(gn)
        optimizer.step()
        param_sigs.append(sum(float(p.sum().item()) for p in model.parameters()))

    return {
        "batch_sigs": batch_sigs, "losses": losses,
        "grad_norms": grad_norms, "param_sigs": param_sigs,
    }
