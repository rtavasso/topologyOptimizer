"""Supervised window construction from a raw trajectory (Section 9).

For history length H and horizon h:
    X_t = {S_{t-H+1}, ..., S_t}   (representation features per node)
    Y_t = T_{t+h}                 (target quantity per node)

We also emit the target-space *history* so history-only baselines (persistence,
EMA, VAR, ...) operate in the target space, and per-node original shapes so
functional/subspace metrics can reconstruct matrices.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from ..logging.reader import TrajectoryReader
from ..topology import representations as reps_mod
from ..utils import linalg as _linalg
import torch


# -- target builders: (reader, name, idx_on_base_grid) -> np.ndarray ----------

def _field_at(reader: TrajectoryReader, key: str, base_step: int):
    if not reader.has_field(key):
        return None
    return reader._slice(key, base_step, None)


def _tgt_future_gradient(reader, name, step):
    return _field_at(reader, f"layer__{name}__G", step)


def _tgt_future_update(reader, name, step):
    return _field_at(reader, f"layer__{name}__DeltaW", step)


def _tgt_future_weight(reader, name, step):
    return _field_at(reader, f"layer__{name}__W", step)


def _tgt_probe_action(reader, name, step):
    return _field_at(reader, f"layer__{name}__probe_WP", step)


def _tgt_gradient_subspace(reader, name, step):
    G = _field_at(reader, f"layer__{name}__G", step)
    rank = min(G.shape)
    key = f"layer__{name}__top_left_singular_vectors_G"
    if reader.has_field(key):
        rank = min(rank, reader.layer_field(name, "top_left_singular_vectors_G", step=0).shape[1])
    U, _, _ = _linalg.top_singular(torch.from_numpy(np.asarray(G)), rank)
    return U.numpy()


TARGET_BUILDERS: Dict[str, Callable] = {
    "future_gradient": _tgt_future_gradient,
    "next_gradient": _tgt_future_gradient,
    "future_update": _tgt_future_update,
    "next_update": _tgt_future_update,
    "future_weight": _tgt_future_weight,
    "next_weight": _tgt_future_weight,
    "probe_action": _tgt_probe_action,
    "future_probe_action": _tgt_probe_action,
    "gradient_subspace": _tgt_gradient_subspace,
}

# targets defined as a difference W_{t+h} - W_t
DELTA_TARGETS = {"weight_delta", "future_weight_delta"}


@dataclass
class WindowTensors:
    feat_hist: np.ndarray      # (N, H, num_nodes, Fnode) sketched features
    tgt_hist: np.ndarray       # (N, H, num_nodes, Tdim) target-space history (padded)
    Y: np.ndarray              # (N, num_nodes, Tdim) target at t+h (padded)
    mask: np.ndarray           # (num_nodes, Tdim) 1 where valid (per node)
    node_shapes: List[tuple]   # original matrix shape per node target
    meta: Dict[str, np.ndarray]
    aux: Optional[Dict[str, np.ndarray]] = None  # auxiliary tensors (analytic baseline)


# matrix-valued gradient/update targets for which the analytic-contraction
# baseline and subspace baselines are meaningful
MATRIX_GRADIENT_TARGETS = {"future_gradient", "next_gradient", "future_update", "next_update"}


# state-dict key -> logged field name
_STATE_FIELDS = {
    "W": "W", "G": "G", "dW": "DeltaW",
    "probe_WP": "probe_WP", "probe_GP": "probe_GP", "probe_prefix": "probe_prefix",
    "gram_in": "gram_in", "gram_out": "gram_out",
    "sv_W": "top_singular_values_W", "sv_G": "top_singular_values_G",
    "sv_dW": "top_singular_values_DeltaW",
    "U_G": "top_left_singular_vectors_G", "V_G": "top_right_singular_vectors_G",
    "act_cov": "activation_covariance", "out_cov": "output_covariance",
    "err_cov": "backprop_error_covariance", "cross_cov": "cross_covariance_error_input",
}

# target name -> logged field used to build it
TARGET_FIELD = {
    "future_gradient": "G", "next_gradient": "G",
    "future_update": "DeltaW", "next_update": "DeltaW",
    "future_weight": "W", "next_weight": "W",
    "weight_delta": "W", "future_weight_delta": "W",
    "probe_action": "probe_WP", "future_probe_action": "probe_WP",
    "gradient_subspace": "G",
}


def _needs_exact_target(target: str) -> bool:
    return target == "gradient_subspace"


def _exact_target_at(reader: TrajectoryReader, target: str, name: str, step: int) -> np.ndarray:
    return TARGET_BUILDERS[target](reader, name, step)


class _RunCache:
    """Preloads every needed field array once per run, then indexes in memory.

    Avoids the per-step, per-field chunked Zarr reads that made windowing
    quadratically slow.
    """

    def __init__(self, reader: TrajectoryReader, fields):
        self.reader = reader
        self.names = reader.layer_names
        self.cad_full = reader.cadence_for(f"layer__{self.names[0]}__W")
        self.store: Dict[str, Dict[str, tuple]] = {}
        for name in self.names:
            self.store[name] = {}
            for field in fields:
                key = f"layer__{name}__{field}"
                if reader.has_field(key):
                    self.store[name][field] = (reader.layer_field(name, field),
                                               reader.cadence_for(key))

    def at(self, name: str, field: str, base_idx: int):
        entry = self.store[name].get(field)
        if entry is None:
            return None
        arr, cad = entry
        idx = (base_idx * self.cad_full) // cad
        if idx >= len(arr):
            idx = len(arr) - 1
        return arr[idx]

    def node_state(self, name: str, base_idx: int, need_whitened: bool = False) -> Dict[str, np.ndarray]:
        state = {sk: self.at(name, f, base_idx) for sk, f in _STATE_FIELDS.items()}
        if need_whitened and state["W"] is not None and state["act_cov"] is not None and state["out_cov"] is not None:
            try:
                Wt = torch.from_numpy(np.asarray(state["W"]))
                Cxx = torch.from_numpy(np.asarray(state["act_cov"]))
                Cyy = torch.from_numpy(np.asarray(state["out_cov"]))
                A = _linalg.matrix_inv_sqrt(Cyy) @ Wt @ _linalg.matrix_sqrt(Cxx)
                state["whitened_W"] = A.numpy()
            except Exception:
                state["whitened_W"] = None
        return state


class _Sketch:
    """Fixed Gaussian sketch to a common node-feature dim (versioned per run)."""

    def __init__(self, out_dim: int, seed: int):
        self.out_dim = out_dim
        self.seed = seed
        self._mats: Dict[int, np.ndarray] = {}

    def apply(self, v: np.ndarray) -> np.ndarray:
        d = v.shape[0]
        if d == self.out_dim:
            return v
        if d not in self._mats:
            rng = np.random.default_rng((self.seed * 1000003) ^ d)
            self._mats[d] = (rng.standard_normal((d, self.out_dim)) / np.sqrt(d)).astype(np.float32)
        return (v @ self._mats[d]).astype(np.float32)


def build_run_windows(
    reader: TrajectoryReader,
    history_length: int,
    horizon: int,
    target: str,
    reps: List[str],
    node_feat_dim: int = 128,
    sketch_seed: int = 0,
    run_index: int = 0,
    max_target_dim: Optional[int] = None,
) -> Optional[WindowTensors]:
    names = reader.layer_names
    base_grid = reader.steps_for(f"layer__{names[0]}__W")
    n_base = len(base_grid)
    cad = reader.cadence_for(f"layer__{names[0]}__W")
    h_idx = max(1, int(np.ceil(horizon / cad)))  # horizon in base-grid units

    sketch = _Sketch(node_feat_dim, sketch_seed)
    scalars = reader.scalars()
    step_to_regime = dict(zip(scalars["step"], scalars.get("data_regime_id", scalars["step"] * 0)))

    if target not in TARGET_FIELD:
        raise ValueError(f"unknown target: {target}")
    is_delta = target in DELTA_TARGETS
    tgt_field = TARGET_FIELD[target]

    # preload everything this (representation, target) needs, once per run
    need_whitened = "R6" in reps
    needed = set(_STATE_FIELDS.values()) | {tgt_field, "W"}
    cache = _RunCache(reader, needed)

    # determine target dims per node from the preloaded target field
    node_target_shapes = []
    for name in names:
        sample = _exact_target_at(reader, target, name, int(base_grid[0])) if _needs_exact_target(target) else cache.at(name, tgt_field, 0)
        node_target_shapes.append(tuple(sample.shape))
    flat_dims = [int(np.prod(s)) for s in node_target_shapes]
    Tdim = max_target_dim or max(flat_dims)

    def tgt_at(name, base_idx):
        if _needs_exact_target(target):
            return _exact_target_at(reader, target, name, int(base_grid[base_idx]))
        return cache.at(name, tgt_field, base_idx)

    # memoize per-(node, step) sketched features and target-space values, since
    # overlapping windows revisit the same steps up to ``history_length`` times.
    _feat_memo: Dict[tuple, np.ndarray] = {}
    _tgt_memo: Dict[tuple, np.ndarray] = {}

    def feat_at(name, hpos):
        k = (name, hpos)
        v = _feat_memo.get(k)
        if v is None:
            st = cache.node_state(name, hpos, need_whitened)
            v = sketch.apply(reps_mod.build_feature(st, reps))
            _feat_memo[k] = v
        return v

    def thist_at(name, hpos):
        k = (name, hpos)
        v = _tgt_memo.get(k)
        if v is None:
            tv = tgt_at(name, hpos)
            if is_delta:
                prev_idx = max(0, hpos - h_idx)
                tv = tv - cache.at(name, "W", prev_idx)
            v = _pad(tv.reshape(-1), Tdim)
            _tgt_memo[k] = v
        return v

    # auxiliary tensors for the analytic deep-linear contraction baseline:
    # per-node W at the anchor step and the end-to-end-gradient history.
    collect_aux = (target in MATRIX_GRADIENT_TARGETS
                   and reader.has_field("network__end_to_end_gradient"))
    if collect_aux:
        gm_arr = reader.network_field("end_to_end_gradient")
        cad_gm = reader.cadence_for("network__end_to_end_gradient")
        w_flat_dims = [int(np.prod(cache.at(n, "W", 0).shape)) for n in names]
        maxWdim = max(w_flat_dims)
        gm_dim = int(np.prod(gm_arr[0].shape))
        gm_shape = tuple(gm_arr[0].shape)

        def gm_at(base_idx):
            idx = (base_idx * cad) // cad_gm
            return gm_arr[min(idx, len(gm_arr) - 1)]

    feat_rows, tgt_hist_rows, Y_rows = [], [], []
    meta_t, meta_phase, meta_regime = [], [], []
    auxW_rows, auxGM_rows = [], []

    last_valid = n_base - h_idx
    for i in range(history_length - 1, last_valid):
        t_target = i + h_idx
        node_feats_H, node_tgt_H = [], []
        for hpos in range(i - history_length + 1, i + 1):
            node_feats_H.append(np.stack([feat_at(name, hpos) for name in names]))
            node_tgt_H.append(np.stack([thist_at(name, hpos) for name in names]))

        Y_nodes = []
        for name in names:
            tv = tgt_at(name, t_target)
            if is_delta:
                tv = tv - cache.at(name, "W", i)
            Y_nodes.append(_pad(tv.reshape(-1), Tdim))

        feat_rows.append(np.stack(node_feats_H))
        tgt_hist_rows.append(np.stack(node_tgt_H))
        Y_rows.append(np.stack(Y_nodes))
        base_t = int(base_grid[i])
        meta_t.append(base_t)
        meta_phase.append(base_t / max(1, reader.total_steps))
        meta_regime.append(int(step_to_regime.get(base_t, 0)))

        if collect_aux:
            auxW_rows.append(np.stack([_pad(cache.at(n, "W", i).reshape(-1), maxWdim) for n in names]))
            auxGM_rows.append(np.stack([_pad(gm_at(hp).reshape(-1), gm_dim)
                                        for hp in range(i - history_length + 1, i + 1)]))

    if not feat_rows:
        return None

    mask = np.zeros((len(names), Tdim), dtype=np.float32)
    for ni, fd in enumerate(flat_dims):
        mask[ni, :fd] = 1.0

    N = len(feat_rows)
    aux = None
    if collect_aux:
        aux = {
            "aux_W": np.stack(auxW_rows).astype(np.float32),          # (N, nodes, maxWdim)
            "aux_GM_hist": np.stack(auxGM_rows).astype(np.float32),    # (N, H, gm_dim)
            "aux_W_shapes": np.array([tuple(cache.at(n, "W", 0).shape) for n in names]),
            "aux_GM_shape": np.array(gm_shape),
        }
    return WindowTensors(
        feat_hist=np.stack(feat_rows).astype(np.float32),
        tgt_hist=np.stack(tgt_hist_rows).astype(np.float32),
        Y=np.stack(Y_rows).astype(np.float32),
        mask=mask,
        node_shapes=node_target_shapes,
        meta={
            "t": np.array(meta_t),
            "phase": np.array(meta_phase, dtype=np.float32),
            "regime": np.array(meta_regime),
            "run_index": np.full(N, run_index),
            "horizon": np.full(N, horizon),
            "history_length": np.full(N, history_length),
            "cadence": np.full(N, cad),
            "h_idx": np.full(N, h_idx),
        },
        aux=aux,
    )


def _pad(v: np.ndarray, dim: int) -> np.ndarray:
    if v.shape[0] == dim:
        return v.astype(np.float32)
    out = np.zeros(dim, dtype=np.float32)
    out[: v.shape[0]] = v
    return out
