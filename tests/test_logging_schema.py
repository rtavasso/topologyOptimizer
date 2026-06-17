"""Logging schema + writer/reader roundtrip (spec Section 8, 20)."""
import tempfile

from conftest import tiny_config

from tad.training.trainer import train_run
from tad.logging.reader import TrajectoryReader
from tad.logging.schema import SCHEMA_VERSION


def test_trajectory_roundtrip():
    cfg = tiny_config("logtest", steps=30)
    rd = train_run(cfg, 0, tempfile.mkdtemp())
    r = TrajectoryReader(rd)

    assert r.attrs["schema_version"] == SCHEMA_VERSION
    assert r.layer_names == ["W1", "W2"]

    # 30 steps logged every step -> exactly 30 rows (no trailing NaN row)
    W = r.layer_field("W1", "W")
    assert W.shape[0] == 30 and W.shape[1:] == (16, 8)
    import numpy as np
    assert np.isfinite(W).all()

    # SVD cadence grid is sparser than the full grid
    sv_steps = r.steps_for("layer__W1__top_singular_values_G")
    assert sv_steps[1] - sv_steps[0] == 5

    s = r.scalars()
    assert "train_loss_before" in s.columns
    assert len(s) == s["step"].nunique() == 30

    # partial loading by step
    one = r.layer_field("W1", "W", step=10)
    assert one.shape == (16, 8)

    # checksums present
    man = r.manifest()
    assert man["schema_version"] == SCHEMA_VERSION
