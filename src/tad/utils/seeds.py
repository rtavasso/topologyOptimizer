"""Deterministic seeding and RNG-state capture for exact replay."""
from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed all RNGs used in the project.

    When ``deterministic`` is set we also force deterministic cuDNN/cuBLAS
    behaviour so that CPU/GPU replay matches within documented tolerances.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # cuBLAS deterministic workspace (no-op on CPU).
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # Strict determinism is enforced only on CPU. On CUDA,
        # use_deterministic_algorithms makes matmul hard-error unless cuBLAS's
        # workspace was configured before the CUDA context initialized — which we
        # can't guarantee in a notebook. GPU runs rely on cuDNN-deterministic +
        # seeded RNG with documented tolerances (spec 20.3).
        if not torch.cuda.is_available():
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass


def derive_seed(base_seed: int, *parts: Any) -> int:
    """Derive a stable child seed from a base seed and arbitrary parts.

    Used to key per-step streaming batches so that an identical batch sequence
    can be regenerated for exact replay and candidate-update evaluation.

    Uses a stable cryptographic hash (NOT Python's builtin ``hash``, which is
    salted per-process by ``PYTHONHASHSEED`` for str/bytes/tuples). This makes
    the derived stream identical across separate ``python`` invocations, which
    is required for exact replay (spec 20.3) and the "identical future batch"
    guarantee for candidate-update evaluation (spec 5.6).
    """
    key = "::".join([str(base_seed)] + [str(p) for p in parts])
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    # Map to a positive 32-bit integer (numpy Generator seed range).
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def make_generator(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


@dataclass
class RngSnapshot:
    """Full RNG state across python/numpy/torch for candidate-update isolation."""

    py_state: Any
    np_state: Any
    torch_state: torch.Tensor
    cuda_state: Any

    @classmethod
    def capture(cls) -> "RngSnapshot":
        cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        return cls(
            py_state=random.getstate(),
            np_state=np.random.get_state(),
            torch_state=torch.get_rng_state(),
            cuda_state=cuda,
        )

    def restore(self) -> None:
        random.setstate(self.py_state)
        np.random.set_state(self.np_state)
        torch.set_rng_state(self.torch_state)
        if self.cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.cuda_state)
