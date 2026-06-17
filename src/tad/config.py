"""Configuration loading, merging, and run provenance.

Configs are plain YAML mapped onto an attribute-accessible ``Config`` object.
Experiment configs may pull in shared sub-configs via an ``includes:`` list and
override fields inline. Every command persists the *resolved* config plus
provenance (git commit, environment, seeds) per Section 19.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml


class Config(Mapping):
    """Recursive, attribute-accessible, read-mostly config wrapper."""

    def __init__(self, data: Dict[str, Any]):
        object.__setattr__(self, "_data", dict(data))

    def __getattr__(self, key: str) -> Any:
        try:
            val = self._data[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Config(val) if isinstance(val, dict) else val

    def __getitem__(self, key):
        val = self._data[key]
        return Config(val) if isinstance(val, dict) else val

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def get(self, key, default=None):
        val = self._data.get(key, default)
        return Config(val) if isinstance(val, dict) else val

    def to_dict(self) -> Dict[str, Any]:
        return _to_plain(self._data)

    def __repr__(self):
        return f"Config({json.dumps(self.to_dict(), default=str)[:200]})"


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, Config):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path: str | Path) -> Config:
    """Load a YAML config, resolving an optional ``includes:`` list.

    Includes are resolved relative to the config file's directory and merged in
    order; the top-level file wins on conflicts.
    """
    path = Path(path)
    raw = load_yaml(path)
    includes: Iterable[str] = raw.pop("includes", []) or []
    merged: Dict[str, Any] = {}
    for inc in includes:
        inc_path = (path.parent / inc).resolve()
        merged = deep_merge(merged, load_config(inc_path).to_dict())
    merged = deep_merge(merged, raw)
    return Config(merged)


@dataclass
class Provenance:
    git_commit: str
    python_version: str
    platform: str
    torch_version: str
    cuda_available: bool
    argv: list

    @classmethod
    def capture(cls) -> "Provenance":
        try:
            import torch

            tv, cuda = torch.__version__, bool(torch.cuda.is_available())
        except Exception:
            tv, cuda = "unavailable", False
        return cls(
            git_commit=_git_commit(),
            python_version=sys.version,
            platform=platform.platform(),
            torch_version=tv,
            cuda_available=cuda,
            argv=list(sys.argv),
        )

    def to_dict(self) -> dict:
        return self.__dict__


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else "not-a-git-repo"
    except Exception:
        return "unavailable"


def save_run_metadata(run_dir: str | Path, config: Config, extra: Dict[str, Any] | None = None) -> None:
    """Persist resolved config + provenance into ``run_dir`` (Section 19)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False)
    meta = {"provenance": Provenance.capture().to_dict(), "env": _safe_env()}
    if extra:
        meta.update(extra)
    with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)


def _safe_env() -> dict:
    keys = ["CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED", "OMP_NUM_THREADS"]
    return {k: os.environ.get(k) for k in keys}
