"""Small autoregressive transformer (Section 6 Phase 3).

Per Section 23 Milestone 7 / Section 6, the transformer extension is begun only
after the linear and immediate-nonlinear experiments establish residual
predictability. The scaffold below records the intended topology composites
(W_Q W_K^T, W_V W_O); the full training path is intentionally deferred.
"""
from __future__ import annotations


class TransformerNotYetEnabled(NotImplementedError):
    pass


def build_transformer(cfg, seed=None):  # pragma: no cover - Phase 3 gate
    raise TransformerNotYetEnabled(
        "Transformer (Phase 3) is gated on completion of E1/E2 per spec Section 23. "
        "Enable by implementing the QKVO logging composites and wiring into TADModel."
    )
