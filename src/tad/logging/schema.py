"""Trajectory schema: field names, cadence classes, validation (Section 8).

Cadence classes determine on which step-grid a field is stored:
  - ``scalar``     : every step, stored in the Parquet metadata table.
  - ``full``       : full tensors, cadence = logging.full_tensor_every.
  - ``svd``        : spectral fields, cadence = logging.svd_every.
  - ``checkpoint`` : full model snapshot, cadence = logging.checkpoint_every.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

SCHEMA_VERSION = 2

# Per-step global scalar fields (Section 8.2).
GLOBAL_SCALARS: List[str] = [
    "run_id", "seed", "step", "epoch", "wall_time",
    "optimizer_type", "learning_rate", "batch_size", "data_regime_id",
    "train_loss_before", "train_loss_after", "validation_loss",
    "gradient_global_norm", "update_global_norm", "parameter_global_norm",
]

# Per-layer full-cadence tensor fields (Section 8.3).
LAYER_FULL_FIELDS: List[str] = [
    "W", "G", "DeltaW", "momentum", "second_moment",
    "activation_mean", "activation_covariance", "output_covariance",
    "backprop_error_mean", "backprop_error_covariance", "cross_covariance_error_input",
    "probe_WP", "probe_GP", "probe_prefix",
    "gram_in", "gram_out",
]

# Per-layer scalar fields stored alongside layer tensors (kept on full grid).
LAYER_SCALAR_FIELDS: List[str] = [
    "weight_frobenius_norm", "gradient_frobenius_norm", "update_frobenius_norm",
    "gradient_momentum_cosine", "effective_rank_W", "stable_rank_W", "condition_number_W",
]

# Per-layer SVD-cadence fields (Section 8.3).
LAYER_SVD_FIELDS: List[str] = [
    "top_singular_values_W", "top_singular_values_G", "top_singular_values_DeltaW",
    "top_left_singular_vectors_G", "top_right_singular_vectors_G",
    "top_left_singular_vectors_DeltaW", "top_right_singular_vectors_DeltaW",
]

# Network-level topology fields (Section 8.4).
NETWORK_FULL_FIELDS: List[str] = [
    "end_to_end_map", "end_to_end_gradient",
]
NETWORK_LIST_FIELDS: List[str] = [
    "prefix_products", "suffix_products", "balance_errors",
    "condition_numbers", "stable_ranks", "effective_ranks",
]


@dataclass
class Cadences:
    full_tensor_every: int = 1
    svd_every: int = 5
    validation_every: int = 25
    checkpoint_every: int = 100

    @classmethod
    def from_config(cls, cfg) -> "Cadences":
        return cls(
            full_tensor_every=int(cfg.get("full_tensor_every", 1)),
            svd_every=int(cfg.get("svd_every", 5)),
            validation_every=int(cfg.get("validation_every", 25)),
            checkpoint_every=int(cfg.get("checkpoint_every", 100)),
        )


def grid_length(total_steps: int, cadence: int) -> int:
    """Number of logged points for steps 0..total_steps-1 on a cadence grid.

    Step s is logged iff s % cadence == 0, so the count is
    floor((total_steps-1)/cadence) + 1. (Sizing to exactly this avoids a
    trailing unwritten NaN row.)
    """
    if total_steps <= 0:
        return 0
    return (total_steps - 1) // cadence + 1


def validate_record(field: str, arr) -> None:
    """Lightweight shape/finiteness validation (Section 8 design principles)."""
    import numpy as np

    if arr is None:
        return
    a = np.asarray(arr)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"non-finite values logged for field '{field}'")
