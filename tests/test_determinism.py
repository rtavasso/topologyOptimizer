"""Data-stream determinism and no train/test leakage (spec Sections 5.6, 20.8)."""
import tempfile
from pathlib import Path

import numpy as np

from conftest import tiny_config
from tad.data import build_data_stream
from tad.datasets.preprocessing import split_runs, _assert_no_leakage


def test_stream_replay_identical_batches():
    cfg = tiny_config()
    s1 = build_data_stream(cfg.data, seed=5, total_steps=50)
    s2 = build_data_stream(cfg.data, seed=5, total_steps=50)
    for step in (0, 7, 13, 49):
        b1, b2 = s1.batch(step), s2.batch(step)
        assert np.allclose(b1.x.numpy(), b2.x.numpy())
        assert np.allclose(b1.y.numpy(), b2.y.numpy())


def test_selection_stream_disjoint():
    cfg = tiny_config()
    s = build_data_stream(cfg.data, seed=5, total_steps=50)
    sel = build_data_stream(cfg.data, seed=5 + 777_777, total_steps=50)
    b, bs = s.batch(3), sel.batch(3)
    assert not np.allclose(b.x.numpy(), bs.x.numpy())


def test_no_leakage_across_splits():
    dirs = [Path(tempfile.mkdtemp()) / f"run_{i}" for i in range(6)]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    split = split_runs(dirs, seed=0)
    # each run appears in exactly one split
    all_names = [d.name for v in split.values() for d in v]
    assert len(all_names) == len(set(all_names))
    _assert_no_leakage(split)
