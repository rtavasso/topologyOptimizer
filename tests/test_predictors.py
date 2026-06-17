"""Tests for the review fixes: cross-process determinism, real subspace
baselines, analytic-contraction baseline, residualized model."""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from conftest import tiny_config, SRC
from tad.training.trainer import train_run
from tad.logging.reader import TrajectoryReader
from tad.datasets.trajectory_windows import build_run_windows
from tad.predictors import build_predictor


def test_derive_seed_stable_across_processes():
    import os
    code = "from tad.utils.seeds import derive_seed; print(derive_seed(5,'batch',3), derive_seed(5,'structures'))"
    def run(hashseed):
        env = {**os.environ, "PYTHONHASHSEED": hashseed, "PYTHONPATH": str(SRC),
               "PYTHONWARNINGS": "ignore"}
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        assert out.returncode == 0, out.stderr
        return out.stdout.strip()
    a, b = run("1"), run("12345")
    assert a == b and a != "", f"derive_seed not stable across processes: {a!r} vs {b!r}"


def _bundle_from_run(rd, target, H=4, h=1):
    r = TrajectoryReader(rd)
    w = build_run_windows(r, history_length=H, horizon=h, target=target,
                          reps=["R1", "R3"], node_feat_dim=16)
    b = {
        "test_feat": w.feat_hist, "test_tgt_hist": w.tgt_hist, "test_Y": w.Y,
        "mask": w.mask, "node_shapes": np.array(w.node_shapes),
        "target": target, "h_idx": int(w.meta["h_idx"][0]),
        "layer_names": np.array(r.layer_names),
    }
    if w.aux is not None:
        for k, v in w.aux.items():
            b[f"test_{k}"] = v
    return b


def test_online_subspace_and_periodic_are_not_persistence():
    cfg = tiny_config("subbase", steps=40)
    rd = train_run(cfg, 0, tempfile.mkdtemp())
    b = _bundle_from_run(rd, "future_gradient", H=8, h=1)
    persist = b["test_tgt_hist"][:, -1]
    for name in ("online_subspace_tracker", "periodic_svd"):
        pred = build_predictor(name, {"rank": 4}).predict(b, "test")
        assert pred.shape == persist.shape
        # genuinely a subspace projection, not a copy of the last gradient
        assert not np.allclose(pred, persist), f"{name} collapsed to persistence"
        assert np.isfinite(pred).all()


def test_analytic_contraction_wiring_is_exact_on_linear():
    """With ema_beta=0 the analytic baseline uses G_M at the anchor step, so the
    contraction G_1=W_2^T G_M, G_2=G_M W_1^T must reproduce the logged layer
    gradients (= persistence) to numerical precision — validates the wiring and
    that the topology aux tensors are correct."""
    cfg = tiny_config("analyt", steps=60)
    rd = train_run(cfg, 0, tempfile.mkdtemp())
    b = _bundle_from_run(rd, "future_gradient", H=8, h=1)
    assert "test_aux_W" in b and "test_aux_GM_hist" in b  # aux emitted for matrix target
    persist = b["test_tgt_hist"][:, -1]  # G at the anchor step
    mask = b["mask"]
    exact = build_predictor("analytic_contraction", {"ema_beta": 0.0}).predict(b, "test")
    err = (((exact - persist) * mask[None]) ** 2).sum() / (((persist * mask[None]) ** 2).sum() + 1e-12)
    assert float(err) < 1e-8, f"contraction wiring not exact: {float(err)}"

    # the default (EMA) variant is a genuine predictor, not a copy of persistence
    default = build_predictor("analytic_contraction", {}).predict(b, "test")
    assert np.isfinite(default).all()
    assert not np.allclose(default, persist)
