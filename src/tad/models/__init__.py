from .base import TADModel, LayerCapture  # noqa: F401
from .deep_linear import DeepLinear  # noqa: F401
from .mlp import NonlinearMLP  # noqa: F401


def build_model(cfg, seed: int | None = None):
    mtype = cfg.get("type", "deep_linear")
    if mtype == "deep_linear":
        return DeepLinear(cfg, seed=seed)
    if mtype in ("mlp", "nonlinear_mlp"):
        return NonlinearMLP(cfg, seed=seed)
    raise ValueError(f"unknown model type: {mtype}")
