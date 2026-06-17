from .baselines import (  # noqa: F401
    Persistence, ConstantVelocity, TunedEMA, Ridge, VectorAutoregression,
    DMDKoopman, OnlineSubspaceTracker, PeriodicSVD, BASELINE_REGISTRY,
)
from .layer_rnn import LayerGRU  # noqa: F401
from .graph_dynamics import TopologyGraphGRU  # noqa: F401
from .subspace_model import SlowSubspaceFastCoordinate  # noqa: F401
from .probabilistic import ConditionalGaussian  # noqa: F401


def build_predictor(name: str, cfg=None):
    cfg = cfg or {}
    name = name.lower()
    if name in BASELINE_REGISTRY:
        return BASELINE_REGISTRY[name](cfg)
    if name in ("layer_gru", "layer_rnn"):
        return LayerGRU(cfg)
    if name in ("topology_graph_gru", "graph_dynamics"):
        return TopologyGraphGRU(cfg, residual=False)
    if name == "residualized_topology_graph_gru":
        return TopologyGraphGRU(cfg, residual=True)
    if name in ("slow_subspace_fast_coordinate", "subspace_model"):
        return SlowSubspaceFastCoordinate(cfg)
    if name in ("conditional_gaussian", "probabilistic"):
        return ConditionalGaussian(cfg)
    raise ValueError(f"unknown predictor: {name}")
