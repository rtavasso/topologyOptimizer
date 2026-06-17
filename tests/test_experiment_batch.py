from pathlib import Path
import importlib.util

import yaml

from tad.config import load_config

_SPEC = importlib.util.spec_from_file_location(
    "run_experiment_batch", Path("scripts/run_experiment_batch.py")
)
_BATCH_RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BATCH_RUNNER)
_command = _BATCH_RUNNER._command


NEXT_CONFIGS = [
    "configs/experiments/e2_colab_relu.yaml",
    "configs/experiments/e2_colab_gelu.yaml",
    "configs/experiments/e3_stochastic_matched.yaml",
    "configs/experiments/e3_largebatch_matched.yaml",
    "configs/experiments/e8_oracle_local.yaml",
    "configs/experiments/e8_oracle_wide.yaml",
]


def test_next_experiment_configs_load():
    for path in NEXT_CONFIGS:
        cfg = load_config(path)
        assert cfg.experiment["name"]
        assert cfg.data["steps"] > 0
        assert cfg.dynamics_dataset["window_stride"] >= 1


def test_next_batch_manifest_is_runnable():
    manifest = yaml.safe_load(Path("configs/experiment_batches/next_after_e1.yaml").read_text())
    names = [e["name"] for e in manifest["experiments"]]
    assert names == [
        "e2_colab_relu",
        "e2_colab_gelu",
        "e3_stochastic_matched",
        "e3_largebatch_matched",
        "e8_oracle_local",
        "e8_oracle_wide",
    ]
    for exp in manifest["experiments"]:
        for stage in exp["stages"]:
            cmd = _command(stage, exp["config"], "cuda")
            assert cmd[2:5] == ["tad.cli", "--device", "cuda"]
