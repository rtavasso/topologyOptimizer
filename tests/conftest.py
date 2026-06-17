import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402

torch.manual_seed(0)


def tiny_config(name="test", steps=30, dims=(8, 16, 4), model_type="deep_linear"):
    from tad.config import Config
    return Config({
        "experiment": {"name": name, "seeds": [0]},
        "data": {"type": "gaussian_matrix_regression", "input_dim": dims[0], "output_dim": dims[-1],
                 "batch_size": 32, "steps": steps,
                 "input_covariance": {"type": "power_law", "alpha": 1.0},
                 "teacher": {"type": "prescribed_spectrum", "rank": 4, "spectrum": "geometric",
                             "condition_number": 10},
                 "noise_std": 0.01, "validation_size": 128},
        "model": {"type": model_type, "dimensions": list(dims), "bias": False,
                  "initialization": "xavier"},
        "optimizer": {"type": "adamw", "learning_rate": 0.01, "betas": [0.9, 0.999],
                      "weight_decay": 0.0},
        "logging": {"full_tensor_every": 1, "svd_every": 5, "validation_every": 10,
                    "checkpoint_every": 20, "svd_rank": 4, "random_probe_count": 4},
    })
