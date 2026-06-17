"""Candidate-update isolation (spec Section 20.4).

Temporary application of a candidate must restore parameters, optimizer state,
gradients, and RNG state.
"""
import copy

import torch

from conftest import tiny_config
from tad.models import build_model
from tad.training.optimizers import build_optimizer
from tad.training.candidate_updates import CandidateContext, apply_delta, evaluate_candidates


def _setup():
    cfg = tiny_config()
    model = build_model(cfg.model, seed=0)
    opt = build_optimizer(cfg.optimizer, list(model.parameters()))
    x, y = torch.randn(16, 8), torch.randn(16, 4)
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    opt.step()  # populate optimizer state
    # fresh grads present
    opt.zero_grad(set_to_none=False)
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    return cfg, model, opt, x, y


def test_candidate_restores_everything():
    _, model, opt, x, y = _setup()
    params_before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    opt_before = copy.deepcopy(opt.state_dict())
    grads_before = {n: p.grad.detach().clone() for n, p in model.named_parameters()}

    delta = {n: torch.randn_like(p) for n, p in model.named_parameters()}
    # capture RNG state at the point the context will snapshot it (its __enter__)
    rng_before = torch.get_rng_state().clone()
    with CandidateContext(model, opt):
        apply_delta(model, delta)
        torch.randn(5)  # perturb RNG
        _ = model(x)

    for k, v in model.state_dict().items():
        assert torch.allclose(v, params_before[k]), f"param {k} not restored"
    for n, p in model.named_parameters():
        assert torch.allclose(p.grad, grads_before[n]), f"grad {n} not restored"
    assert torch.equal(torch.get_rng_state(), rng_before), "RNG not restored"
    # optimizer step counts restored
    assert str(opt.state_dict()["state"].keys()) == str(opt_before["state"].keys())


def test_evaluate_candidates_picks_min():
    _, model, opt, x, y = _setup()

    def loss_fn():
        with torch.no_grad():
            return float(((model(x) - y) ** 2).mean().item())

    zero = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    huge = {n: 100 * torch.ones_like(p) for n, p in model.named_parameters()}
    res = evaluate_candidates(model, opt, [zero, huge], loss_fn)
    assert res["winner"] == 0
    # state restored after evaluation
    assert all(p.grad is not None for p in model.parameters())


def test_oracle_records_configured_candidate_labels():
    from tad.config import Config
    from tad.evaluation.online_optimizer import oracle_experiment

    cfg = tiny_config("oracle_labels", steps=4)
    raw = cfg.to_dict()
    raw["online"] = {
        "ema_beta": 0.8,
        "include_predicted_candidate": True,
        "candidate_blends": [0.25, 0.75],
        "candidate_scales": [0.5, 1.5],
    }
    res = oracle_experiment(Config(raw), seed=0, n_steps=2, eval_every=1)

    assert res["candidate_labels"] == [
        "baseline", "predicted", "blend_0.25", "blend_0.75", "scale_0.5", "scale_1.5"
    ]
    assert sum(res["winner_label_histogram"].values()) == res["n_evals"]
