"""Exact product update test (spec Section 20.2).

M_{t+1} - M_t = dW2 W1 + W2 dW1 + dW2 dW1.
"""
import torch

from tad.models import build_model
from tad.config import Config
from tad.topology.products import product_update_decomposition, end_to_end_map


def test_product_update_decomposition():
    cfg = Config({"type": "deep_linear", "dimensions": [6, 5, 4], "bias": False,
                  "initialization": "gaussian"})
    model = build_model(cfg, seed=1)
    W1 = model.weight("W1").detach().clone()
    W2 = model.weight("W2").detach().clone()
    M_before = end_to_end_map([W1, W2])

    dW1 = 0.01 * torch.randn_like(W1)
    dW2 = 0.01 * torch.randn_like(W2)
    with torch.no_grad():
        model.weight("W1").add_(dW1)
        model.weight("W2").add_(dW2)
    M_after = end_to_end_map([model.weight("W1").detach(), model.weight("W2").detach()])

    t1, t2, t3 = product_update_decomposition(W2, W1, dW2, dW1)
    assert torch.allclose(M_after - M_before, t1 + t2 + t3, atol=1e-6)
