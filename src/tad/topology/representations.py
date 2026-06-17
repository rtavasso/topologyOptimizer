"""State representations R0-R7 (Section 10).

Each builder turns a per-layer logged state (a dict of numpy arrays for one
step) into a flat feature vector. The dynamics dataset stacks these across the
history window. R7 (learned latent) is produced inside the predictor, not here.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

EPS = 1e-8


def _flat(*arrs) -> np.ndarray:
    return np.concatenate([np.asarray(a, dtype=np.float32).reshape(-1) for a in arrs if a is not None])


def r0_raw(state: Dict[str, np.ndarray]) -> np.ndarray:
    """Raw parameters: W, G, M-related entries."""
    return _flat(state.get("W"), state.get("G"))


def r1_delta(state: Dict[str, np.ndarray]) -> np.ndarray:
    """Delta representation: dW, G."""
    return _flat(state.get("dW"), state.get("G"))


def r2_probe(state: Dict[str, np.ndarray]) -> np.ndarray:
    """Probe-action representation: W P, G P."""
    return _flat(state.get("probe_WP"), state.get("probe_GP"))


def r3_spectral(state: Dict[str, np.ndarray]) -> np.ndarray:
    """Spectral: top singular values + flattened top vectors for W, G, dW."""
    return _flat(
        state.get("sv_W"), state.get("sv_G"), state.get("sv_dW"),
        state.get("U_G"), state.get("V_G"),
    )


def r4_covariance(state: Dict[str, np.ndarray]) -> np.ndarray:
    """Covariance / functional: input, output, error covariances + cross-cov."""
    return _flat(
        state.get("act_cov"), state.get("out_cov"),
        state.get("err_cov"), state.get("cross_cov"),
    )


def r5_topology(state: Dict[str, np.ndarray]) -> np.ndarray:
    """Topology-aware product: W, gram(W^T W), gram(W W^T), probe of prefix."""
    return _flat(state.get("W"), state.get("gram_in"), state.get("gram_out"),
                 state.get("probe_prefix"))


def r6_whitened(state: Dict[str, np.ndarray]) -> np.ndarray:
    """Whitened operator A = Cyy^{-1/2} W Cxx^{1/2} (precomputed at build time)."""
    return _flat(state.get("whitened_W"))


REGISTRY = {
    "R0": r0_raw,
    "R1": r1_delta,
    "R2": r2_probe,
    "R3": r3_spectral,
    "R4": r4_covariance,
    "R5": r5_topology,
    "R6": r6_whitened,
}


def build_feature(state: Dict[str, np.ndarray], reps: List[str]) -> np.ndarray:
    vecs = [REGISTRY[r](state) for r in reps]
    vecs = [v for v in vecs if v.size > 0]
    if not vecs:
        return np.zeros(1, dtype=np.float32)
    return np.concatenate(vecs)
