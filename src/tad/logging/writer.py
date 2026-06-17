"""Trajectory writer (Section 8.1).

Tensor fields go to a chunked Zarr group (partial loading by field/step/layer),
scalars to a Parquet table, probes/config to JSON. Full checkpoints use
safetensors. Lazily creates Zarr arrays sized by each field's cadence grid; the
logger never mutates training (only ``.detach()``ed copies are stored).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import zarr

from .schema import Cadences, SCHEMA_VERSION, grid_length, validate_record


def _key(scope: str, *parts: str) -> str:
    return "__".join([scope, *parts])


class TrajectoryWriter:
    def __init__(self, run_dir, run_id: str, seed: int, total_steps: int,
                 cadences: Cadences, layer_names: List[str], probes=None,
                 attrs: Optional[dict] = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.seed = seed
        self.total_steps = total_steps
        self.cad = cadences
        self.layer_names = layer_names

        self.store_path = str(self.run_dir / "tensors.zarr")
        self.root = zarr.open_group(self.store_path, mode="w")
        self.root.attrs["schema_version"] = SCHEMA_VERSION
        self.root.attrs["run_id"] = run_id
        self.root.attrs["seed"] = int(seed)
        self.root.attrs["total_steps"] = int(total_steps)
        self.root.attrs["layer_names"] = list(layer_names)
        self.root.attrs["cadences"] = {
            "full_tensor_every": cadences.full_tensor_every,
            "svd_every": cadences.svd_every,
            "validation_every": cadences.validation_every,
            "checkpoint_every": cadences.checkpoint_every,
        }
        if attrs:
            for k, v in attrs.items():
                self.root.attrs[k] = v

        self._arrays: Dict[str, zarr.Array] = {}
        self._field_cadence: Dict[str, int] = {}
        # one merged row per step (multiple callers contribute columns)
        self._scalar_rows: Dict[int, dict] = {}
        # time-contiguous write buffers (avoid one chunk-file per timestep)
        self._chunk_rows = 64
        self._buffers: Dict[str, list] = {}
        self._buf_start: Dict[str, int] = {}

        if probes is not None:
            np.savez(self.run_dir / "probes.npz", **probes.to_numpy())

    # -- array management ----------------------------------------------------
    def _get_array(self, key: str, cadence: int, sample: np.ndarray) -> zarr.Array:
        if key not in self._arrays:
            glen = grid_length(self.total_steps, cadence)
            shape = (glen, *sample.shape)
            crows = min(self._chunk_rows, max(1, glen))
            chunks = (crows, *sample.shape)
            arr = self.root.create_array(name=key, shape=shape, chunks=chunks,
                                         dtype="float32", fill_value=float("nan"))
            self._arrays[key] = arr
            self._field_cadence[key] = cadence
            self._buffers[key] = []
            self._buf_start[key] = 0
        return self._arrays[key]

    def _flush(self, key: str) -> None:
        buf = self._buffers.get(key)
        if not buf:
            return
        za = self._arrays[key]
        start = self._buf_start[key]
        block = np.stack(buf)
        end = min(start + len(buf), za.shape[0])
        za[start:end] = block[: end - start]
        self._buf_start[key] = end
        self._buffers[key] = []

    def _write(self, key: str, step: int, cadence: int, tensor) -> None:
        if tensor is None:
            return
        arr = np.asarray(tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor,
                         dtype=np.float32)
        validate_record(key, arr)
        self._get_array(key, cadence, arr)
        self._buffers[key].append(arr)
        if len(self._buffers[key]) >= self._chunk_rows:
            self._flush(key)

    # -- public logging API --------------------------------------------------
    def log_scalars(self, step: int, row: dict) -> None:
        bucket = self._scalar_rows.setdefault(step, {"step": step})
        for k, v in row.items():
            # don't let a later None overwrite a real value
            if v is not None or k not in bucket:
                bucket[k] = v

    def log_layer_full(self, step: int, name: str, field: str, tensor) -> None:
        self._write(_key("layer", name, field), step, self.cad.full_tensor_every, tensor)

    def log_layer_svd(self, step: int, name: str, field: str, tensor) -> None:
        self._write(_key("layer", name, field), step, self.cad.svd_every, tensor)

    def log_network_full(self, step: int, field: str, tensor) -> None:
        self._write(_key("network", field), step, self.cad.full_tensor_every, tensor)

    def checkpoint(self, step: int, model) -> None:
        from safetensors.torch import save_file

        ckpt_dir = self.run_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
        save_file(state, str(ckpt_dir / f"step_{step:07d}.safetensors"))

    # -- finalize ------------------------------------------------------------
    def close(self) -> dict:
        for key in list(self._arrays.keys()):
            self._flush(key)
        rows = [self._scalar_rows[s] for s in sorted(self._scalar_rows)]
        df = pd.DataFrame(rows)
        scalar_path = self.run_dir / "scalars.parquet"
        df.to_parquet(scalar_path, index=False)

        # record field->cadence map and checksums for validation
        self.root.attrs["field_cadence"] = self._field_cadence
        checks = self._checksums()
        with open(self.run_dir / "checksums.json", "w", encoding="utf-8") as f:
            json.dump(checks, f, indent=2)
        manifest = {
            "run_id": self.run_id, "seed": int(self.seed),
            "total_steps": int(self.total_steps),
            "tensor_fields": sorted(self._arrays.keys()),
            "scalar_columns": list(df.columns),
            "schema_version": SCHEMA_VERSION,
        }
        with open(self.run_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return manifest

    def _checksums(self) -> dict:
        out = {}
        for key, arr in self._arrays.items():
            h = hashlib.sha1()
            data = np.asarray(arr[...])
            h.update(np.nan_to_num(data).tobytes())
            out[key] = h.hexdigest()[:16]
        return out
