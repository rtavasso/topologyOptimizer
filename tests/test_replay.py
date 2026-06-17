"""Replay determinism (spec Section 20.3)."""
from conftest import tiny_config

from tad.training.replay import replay_run


def test_replay_is_deterministic():
    cfg = tiny_config("replay", steps=25)
    a = replay_run(cfg, seed=3, n_steps=25)
    b = replay_run(cfg, seed=3, n_steps=25)
    assert a["batch_sigs"] == b["batch_sigs"], "batch sequence differs"
    assert a["losses"] == b["losses"], "losses differ"
    assert a["grad_norms"] == b["grad_norms"], "gradients differ"
    assert a["param_sigs"] == b["param_sigs"], "parameters differ"


def test_different_seed_differs():
    cfg = tiny_config("replay", steps=15)
    a = replay_run(cfg, seed=1, n_steps=15)
    b = replay_run(cfg, seed=2, n_steps=15)
    assert a["batch_sigs"] != b["batch_sigs"]
