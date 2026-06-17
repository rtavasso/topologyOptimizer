"""Analytic gradient test (spec Section 20.1).

G_2 = G_M W_1^T,  G_1 = W_2^T G_M  for a two-layer linear network, compared to
PyTorch autograd.
"""
import torch

from tad.models import build_model
from tad.config import Config


def test_two_layer_linear_gradient_contractions():
    cfg = Config({"type": "deep_linear", "dimensions": [6, 5, 4], "bias": False,
                  "initialization": "gaussian"})
    model = build_model(cfg, seed=0)
    x = torch.randn(32, 6)
    y = torch.randn(32, 4)

    pred = model(x)
    loss = ((pred - y) ** 2).sum(1).mean() / pred.shape[1]
    loss.backward()

    W1 = model.weight("W1").detach()
    W2 = model.weight("W2").detach()
    G1 = model.weight("W1").grad.detach()
    G2 = model.weight("W2").grad.detach()

    # end-to-end gradient G_M from the MSE: dL/dM = (2/(B*dy)) (Mx - y) x^T
    M = W2 @ W1
    B, dy = x.shape[0], y.shape[1]
    G_M = (2.0 / (B * dy)) * (x @ M.T - y).T @ x

    assert torch.allclose(G2, G_M @ W1.T, atol=1e-5), "G2 != G_M W1^T"
    assert torch.allclose(G1, W2.T @ G_M, atol=1e-5), "G1 != W2^T G_M"


def test_logged_trajectory_satisfies_contractions():
    """The logged (W_t, G_t, G_M) must satisfy the contractions (Sections 13.11,
    20.1) — guards W_t/G_t step-synchronization in the logger."""
    import tempfile
    from conftest import tiny_config
    from tad.training.trainer import train_run
    from tad.logging.reader import TrajectoryReader
    from tad.evaluation.temporal_structure import analytic_contraction_check

    cfg = tiny_config("contr", steps=20)
    rd = train_run(cfg, 0, tempfile.mkdtemp())
    res = analytic_contraction_check(TrajectoryReader(rd))
    assert res["G1_contraction_nmse"] < 1e-8, res
    assert res["G2_contraction_nmse"] < 1e-8, res
