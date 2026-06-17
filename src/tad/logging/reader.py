"""Trajectory reader with partial loading (Section 8.1).

Supports loading by run, field, step range, and layer. Tensor fields stored on a
cadence grid expose ``steps_for(field)`` so callers can align sparser fields
(e.g. SVD every 5) onto a denser base grid.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import zarr


class TrajectoryReader:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.root = zarr.open_group(str(self.run_dir / "tensors.zarr"), mode="r")
        self.attrs = dict(self.root.attrs)
        self.cadences = self.attrs.get("cadences", {})
        self.field_cadence = self.attrs.get("field_cadence", {})
        self.layer_names: List[str] = list(self.attrs.get("layer_names", []))
        self.total_steps = int(self.attrs.get("total_steps", 0))
        self._scalars: Optional[pd.DataFrame] = None

    # -- metadata ------------------------------------------------------------
    @property
    def run_id(self) -> str:
        return str(self.attrs.get("run_id"))

    @property
    def seed(self) -> int:
        return int(self.attrs.get("seed"))

    def field_keys(self) -> List[str]:
        return list(self.root.array_keys())

    def manifest(self) -> dict:
        with open(self.run_dir / "manifest.json", "r", encoding="utf-8") as f:
            return json.load(f)

    # -- scalars -------------------------------------------------------------
    def scalars(self) -> pd.DataFrame:
        if self._scalars is None:
            self._scalars = pd.read_parquet(self.run_dir / "scalars.parquet")
        return self._scalars

    # -- tensor access -------------------------------------------------------
    def _key(self, scope, *parts):
        return "__".join([scope, *parts])

    def cadence_for(self, key: str) -> int:
        return int(self.field_cadence.get(key, 1))

    def steps_for(self, key: str) -> np.ndarray:
        cad = self.cadence_for(key)
        n = self.root[key].shape[0]
        return np.arange(n) * cad

    def layer_field(self, name: str, field: str, step: Optional[int] = None,
                    step_range: Optional[tuple] = None) -> np.ndarray:
        key = self._key("layer", name, field)
        return self._slice(key, step, step_range)

    def network_field(self, field: str, step: Optional[int] = None,
                      step_range: Optional[tuple] = None) -> np.ndarray:
        key = self._key("network", field)
        return self._slice(key, step, step_range)

    def has_field(self, key: str) -> bool:
        return key in set(self.root.array_keys())

    def _slice(self, key, step, step_range):
        arr = self.root[key]
        cad = self.cadence_for(key)
        if step is not None:
            return np.asarray(arr[step // cad])
        if step_range is not None:
            lo, hi = step_range
            return np.asarray(arr[lo // cad: (hi + cad - 1) // cad])
        return np.asarray(arr[...])

    def probes(self) -> dict:
        path = self.run_dir / "probes.npz"
        if not path.exists():
            return {}
        return dict(np.load(path))
