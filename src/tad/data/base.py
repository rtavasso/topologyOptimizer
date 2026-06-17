"""Base data-stream interface and the per-step batch container."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class Batch:
    """A single training batch plus its ground-truth regime label (Section 5.5)."""

    x: torch.Tensor  # (B, in_dim)
    y: torch.Tensor  # (B, out_dim)
    regime_id: int
    step: int


class DataStream(Protocol):
    """A reproducible per-step data source.

    Implementations must produce identical batches given the same (seed, step)
    so that exact replay and candidate-update evaluation are possible
    (Section 5.6).
    """

    input_dim: int
    output_dim: int

    def batch(self, step: int) -> Batch:
        ...

    def validation_set(self) -> Batch:
        ...
