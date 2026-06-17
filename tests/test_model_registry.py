import pytest

from tad.config import Config
from tad.models import build_model
from tad.models.transformer import TransformerNotYetEnabled


def test_residual_linear_model_type_is_registered():
    cfg = Config({"type": "residual_linear", "dimensions": [4, 4, 4], "bias": False})
    model = build_model(cfg, seed=0)

    assert type(model).__name__ == "ResidualLinear"
    assert model.layer_names == ["W1", "W2"]


def test_transformer_model_type_reaches_phase_gate():
    cfg = Config({"type": "transformer"})

    with pytest.raises(TransformerNotYetEnabled):
        build_model(cfg, seed=0)
