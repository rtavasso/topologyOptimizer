"""Training loop with dense trajectory logging (Sections 4.1, 8, 16 E0/E1)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import time

import numpy as np
import torch
import torch.nn.functional as F

from ..config import Config, save_run_metadata
from ..data import build_data_stream
from ..models import build_model
from ..logging import TrajectoryWriter, build_probes
from ..logging.schema import Cadences
from ..utils.seeds import set_global_seed
from ..utils.profiling import ComputeLedger, matmul_flops
from .optimizers import build_optimizer
from .hooks import LoggingHook


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-spec base loss: mean over batch of (1/dy) ||yhat - y||^2 (Section 4.1)."""
    return ((pred - target) ** 2).sum(dim=1).mean() / pred.shape[1]


def global_norm(tensors) -> float:
    sq = sum(float((t.detach() ** 2).sum().item()) for t in tensors if t is not None)
    return sq ** 0.5


def train_run(cfg: Config, seed: int, out_dir, device: str = "cpu",
              full_batch: bool = False) -> Path:
    set_global_seed(seed)
    out_dir = Path(out_dir)
    run_id = f"{cfg.experiment['name']}__seed{seed}"
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg.data
    total_steps = int(data_cfg.get("steps", 5000))
    stream = build_data_stream(data_cfg, seed, total_steps, device=device)
    model = build_model(cfg.model, seed=seed).to(device)
    model.enable_capture(True)
    is_linear = cfg.model.get("type", "deep_linear") == "deep_linear"

    optimizer = build_optimizer(cfg.optimizer, list(model.parameters()))
    cadences = Cadences.from_config(cfg.logging)
    probes = build_probes(model, stream, cfg.logging, seed, device=device)

    writer = TrajectoryWriter(
        run_dir=run_dir, run_id=run_id, seed=seed, total_steps=total_steps,
        cadences=cadences, layer_names=model.layer_names, probes=probes,
        attrs={"optimizer_type": cfg.optimizer.get("type", "adamw"),
               "model_type": cfg.model.get("type", "deep_linear"),
               "is_linear": is_linear, "full_batch": full_batch},
    )
    hook = LoggingHook(model, writer, probes, optimizer, cadences,
                       svd_rank=int(cfg.logging.get("svd_rank", 16)), is_linear=is_linear)
    ledger = ComputeLedger()

    lr = float(cfg.optimizer.get("learning_rate", 1e-3))
    bs = int(data_cfg.get("batch_size", 256))
    val = stream.validation_set()
    t0 = time.perf_counter()

    for step in range(total_steps):
        batch = stream.batch(step)
        pre_weights = {n: model.weight(n).detach().clone() for n in model.layer_names}

        optimizer.zero_grad(set_to_none=True)
        with ledger.timer("forward_backward"):
            pred = model(batch.x)
            loss = mse_loss(pred, batch.y)
            loss.backward()
        train_loss_before = float(loss.item())

        gnorm = global_norm([p.grad for p in model.parameters()])
        pnorm = global_norm(list(model.parameters()))

        with ledger.timer("optimizer_step"):
            optimizer.step()
        unorm = global_norm([model.weight(n) - pre_weights[n] for n in model.layer_names])

        # rough FLOP accounting (dense matmul, fwd+bwd ~ 3x forward)
        ledger.add_flops("forward_backward", 3 * _forward_flops(model, batch.x.shape[0]))
        ledger.add_flops("optimizer_step", sum(p.numel() for p in model.parameters()) * 4)

        val_loss = None
        if step % cadences.validation_every == 0:
            with torch.no_grad():
                model.enable_capture(False)
                val_loss = float(mse_loss(model(val.x), val.y).item())
                model.enable_capture(True)

        writer.log_scalars(step, {
            "run_id": run_id, "seed": seed, "epoch": step * bs // max(1, int(data_cfg.get("train_size", total_steps * bs))),
            "wall_time": time.perf_counter() - t0,
            "optimizer_type": cfg.optimizer.get("type", "adamw"),
            "learning_rate": lr, "batch_size": bs, "data_regime_id": batch.regime_id,
            "train_loss_before": train_loss_before, "train_loss_after": None,
            "validation_loss": val_loss,
            "gradient_global_norm": gnorm, "update_global_norm": unorm,
            "parameter_global_norm": pnorm,
        })
        hook.on_step(step, pre_weights, {})

        if step % cadences.checkpoint_every == 0:
            writer.checkpoint(step, model)

    writer.checkpoint(total_steps - 1, model)
    manifest = writer.close()
    import hashlib
    teacher_fp = hashlib.sha1(
        np.ascontiguousarray(stream.teacher_matrix(0.0, 0), dtype=np.float64).tobytes()
    ).hexdigest()
    save_run_metadata(run_dir, cfg, extra={"manifest": manifest, "compute": ledger.to_dict(),
                                           "device": device, "full_batch": full_batch,
                                           "seed": int(seed), "teacher_fingerprint": teacher_fp})
    return run_dir


def _forward_flops(model, batch_size: int) -> float:
    total = 0
    for n in model.layer_names:
        out_d, in_d = model.weight(n).shape
        total += matmul_flops(batch_size, in_d, out_d)
    return total


class Trainer:
    """Thin object wrapper for programmatic use / tests."""

    def __init__(self, cfg: Config, seed: int, device: str = "cpu"):
        self.cfg, self.seed, self.device = cfg, seed, device

    def run(self, out_dir, full_batch: bool = False) -> Path:
        return train_run(self.cfg, self.seed, out_dir, self.device, full_batch)
