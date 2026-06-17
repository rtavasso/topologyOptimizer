#!/usr/bin/env python
"""Run a manifest-defined batch of TAD experiments.

The manifest is intentionally simple: each experiment has a config path and a
list of TAD CLI stages. This keeps Colab and local runs reproducible without
adding a second experiment framework.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


CONFIG_STAGES = {
    "generate-trajectories",
    "build-dataset",
    "train-predictor",
    "evaluate-predictor",
    "run-online-optimizer",
    "run-e1",
}


def _command(stage: str, config: str, device: str) -> list[str]:
    if stage not in CONFIG_STAGES:
        raise ValueError(f"unsupported batch stage: {stage}")
    return [sys.executable, "-m", "tad.cli", "--device", device, stage, "--config", config]


def run_batch(manifest_path: Path, device: str, dry_run: bool = False) -> None:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    experiments = manifest.get("experiments", [])
    if not experiments:
        raise ValueError(f"no experiments in {manifest_path}")

    for exp in experiments:
        name = exp["name"]
        config = exp["config"]
        stages = exp.get("stages", ["run-e1"])
        print(f"== {name} ==")
        print(exp.get("question", ""))
        for stage in stages:
            cmd = _command(stage, config, device)
            print("+", " ".join(cmd), flush=True)
            if not dry_run:
                subprocess.run(cmd, check=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run a TAD experiment batch manifest.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    run_batch(args.manifest, args.device, args.dry_run)


if __name__ == "__main__":
    main()
