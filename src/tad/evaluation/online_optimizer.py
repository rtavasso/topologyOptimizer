"""From prediction to optimization (Section 14).

Implements the Tier-II oracle candidate-update selection (14.1) and the
predicted-subspace optimizer comparison (14.2/E9). Candidate evaluation always
uses a *held-out* selection microbatch (interpretation rule 4) and full state
isolation (Section 20.4). A learned predictor can be supplied via
``predictor_delta_fn``; by default a strong online proxy (EMA-gradient /
EMA-subspace projection) stands in so the oracle question — do better local
moves exist in the proposal family? — can be answered without the full offline
pipeline wired online.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from ..config import Config
from ..data import build_data_stream
from ..models import build_model
from ..training.optimizers import build_optimizer
from ..training.candidate_updates import CandidateContext, apply_delta, evaluate_candidates
from ..training.trainer import mse_loss
from ..utils import linalg
from ..utils.seeds import set_global_seed
from ..utils.profiling import ComputeLedger, matmul_flops


def _named_param_deltas(model, before: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {n: (p.detach() - before[n]) for n, p in model.named_parameters()}


def _baseline_delta(model, optimizer) -> Dict[str, torch.Tensor]:
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    with CandidateContext(model, optimizer):
        optimizer.step()
        delta = {n: (p.detach().clone() - before[n]) for n, p in model.named_parameters()}
    return delta


def _scale(delta, alpha):
    return {n: alpha * d for n, d in delta.items()}


def _blend(d1, d2, alpha):
    return {n: alpha * d1[n] + (1 - alpha) * d2.get(n, torch.zeros_like(d1[n])) for n in d1}


def oracle_experiment(cfg: Config, seed: int, n_steps: int = 400, eval_every: int = 5,
                      device: str = "cpu",
                      predictor_delta_fn: Optional[Callable] = None) -> dict:
    set_global_seed(seed)
    total_steps = int(cfg.data.get("steps", n_steps))
    stream = build_data_stream(cfg.data, seed, total_steps, device=device)
    model = build_model(cfg.model, seed=seed).to(device)
    optimizer = build_optimizer(cfg.optimizer, list(model.parameters()))
    lr = float(cfg.optimizer.get("learning_rate", 1e-3))
    ledger = ComputeLedger()

    g_ema = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    beta = 0.9
    records: List[dict] = []
    # held-out selection stream uses a disjoint key
    sel_stream = build_data_stream(cfg.data, seed + 777_777, total_steps, device=device)

    for step in range(n_steps):
        batch = stream.batch(step)
        optimizer.zero_grad(set_to_none=True)
        loss = mse_loss(model(batch.x), batch.y)
        loss.backward()
        for n, p in model.named_parameters():
            g_ema[n] = beta * g_ema[n] + (1 - beta) * p.grad.detach()

        if step % eval_every == 0:
            d_base = _baseline_delta(model, optimizer)               # Delta_1
            if predictor_delta_fn is not None:
                d_pred = predictor_delta_fn(model, optimizer, g_ema, lr)
            else:
                d_pred = {n: -lr * g_ema[n] for n, p in model.named_parameters()}  # proxy Delta_2
            candidates = [
                d_base,
                d_pred,
                _blend(d_base, d_pred, 0.5),                          # Delta_3
                _scale(d_base, 0.5),
                _scale(d_base, 2.0),
            ]
            sel = sel_stream.batch(step)
            def sel_loss():
                with torch.no_grad():
                    return float(mse_loss(model(sel.x), sel.y).item())
            res = evaluate_candidates(model, optimizer, candidates, sel_loss)
            # same-batch vs held-out gap (interpretation rule 4)
            def train_loss():
                with torch.no_grad():
                    return float(mse_loss(model(batch.x), batch.y).item())
            res_same = evaluate_candidates(model, optimizer, candidates, train_loss)
            res["winner_same_batch"] = res_same["winner"]
            res["same_vs_heldout_gap"] = res_same["improvement_over_baseline"] - res["improvement_over_baseline"]
            res["step"] = step
            res["n_candidates"] = len(candidates)
            ledger.add_flops("candidate_eval", len(candidates) * matmul_flops(
                sel.x.shape[0], stream.input_dim, stream.output_dim))
            records.append(res)

        optimizer.step()  # real step continues training (Delta_1)

    winners = np.array([r["winner"] for r in records])
    improvements = np.array([r["improvement_over_baseline"] for r in records])
    return {
        "n_evals": len(records),
        "baseline_win_rate": float(np.mean(winners == 0)),
        "predicted_win_rate": float(np.mean(winners == 1)),
        "mean_improvement_over_baseline": float(np.mean(improvements)),
        "frac_positive_improvement": float(np.mean(improvements > 1e-9)),
        "winner_histogram": {int(k): int(v) for k, v in zip(*np.unique(winners, return_counts=True))},
        "mean_same_vs_heldout_gap": float(np.mean([r["same_vs_heldout_gap"] for r in records])),
        "compute": ledger.to_dict(),
        "records": records,
    }


def subspace_optimizer_experiment(cfg: Config, seed: int, n_steps: int = 400,
                                  rank: int = 8, device: str = "cpu") -> dict:
    """Compare full update vs EMA-subspace-projected update vs prev-SVD subspace.

    All variants share the exact same data stream (paired comparison, 15.4).
    """
    methods = ["full", "ema_subspace", "prev_svd", "ema_subspace_plus_residual"]
    curves: Dict[str, List[float]] = {}
    for method in methods:
        set_global_seed(seed)
        total_steps = int(cfg.data.get("steps", n_steps))
        stream = build_data_stream(cfg.data, seed, total_steps, device=device)
        model = build_model(cfg.model, seed=seed).to(device)
        optimizer = build_optimizer(cfg.optimizer, list(model.parameters()))
        val = stream.validation_set()
        g_ema = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        prev_g = {n: None for n, p in model.named_parameters()}
        beta = 0.9
        vlosses = []
        for step in range(n_steps):
            batch = stream.batch(step)
            optimizer.zero_grad(set_to_none=True)
            loss = mse_loss(model(batch.x), batch.y)
            loss.backward()
            for n, p in model.named_parameters():
                if p.grad is None or p.ndim != 2:
                    continue
                G = p.grad.detach()
                g_ema[n] = beta * g_ema[n] + (1 - beta) * G
                if method == "full":
                    pass
                else:
                    if method == "ema_subspace" or method == "ema_subspace_plus_residual":
                        src = g_ema[n]
                    else:  # prev_svd
                        src = prev_g[n] if prev_g[n] is not None else G
                    U, S, V = linalg.top_singular(src, min(rank, *G.shape))
                    G_proj = U @ (U.transpose(0, 1) @ G @ V) @ V.transpose(0, 1)
                    if method == "ema_subspace_plus_residual":
                        G_proj = G_proj + 0.1 * (G - G_proj)
                    p.grad.copy_(G_proj)
                prev_g[n] = G.clone()
            optimizer.step()
            if step % 25 == 0:
                with torch.no_grad():
                    vlosses.append(float(mse_loss(model(val.x), val.y).item()))
        curves[method] = vlosses
    return {"curves": curves, "rank": rank, "n_steps": n_steps}
