"""SVD sign/subspace invariance (spec Section 20.5)."""
import torch

from conftest import tiny_config
from tad.utils import linalg


def test_subspace_overlap_sign_invariant():
    A = torch.randn(10, 6)
    U, S, V = linalg.top_singular(A, 4)
    # flip signs of some singular vectors
    flip = torch.tensor([1.0, -1.0, 1.0, -1.0])
    U2 = U * flip.unsqueeze(0)
    assert abs(linalg.subspace_overlap(U, U2) - 1.0) < 1e-5
    assert linalg.projection_distance(U, U2) < 1e-5


def test_energy_capture_full_rank_is_one():
    A = torch.randn(8, 5)
    U, S, V = linalg.top_singular(A, 5)
    cap = linalg.energy_capture(U, A, V)
    assert abs(cap - 1.0) < 1e-4


def test_effective_and_stable_rank_bounds():
    A = torch.eye(6)
    assert abs(linalg.effective_rank(A) - 6.0) < 1e-4
    assert abs(linalg.stable_rank(A) - 6.0) < 1e-4
    low = torch.zeros(6, 6); low[0, 0] = 5.0
    assert linalg.effective_rank(low) < 1.5
    assert linalg.stable_rank(low) < 1.5


def test_sign_alignment_deterministic():
    A = torch.randn(7, 7)
    U1, _, V1 = linalg.top_singular(A, 3)
    U2, _, V2 = linalg.top_singular(A, 3)
    assert torch.allclose(U1, U2)
    assert torch.allclose(V1, V2)


def test_gradient_subspace_target_uses_exact_horizon_with_sparse_svd():
    import tempfile

    from tad.datasets.trajectory_windows import build_run_windows
    from tad.logging.reader import TrajectoryReader
    from tad.training.trainer import train_run

    cfg = tiny_config("subspace_horizon", steps=8)
    raw = cfg.to_dict()
    raw["logging"]["svd_every"] = 5
    cfg = type(cfg)(raw)
    rd = train_run(cfg, 0, tempfile.mkdtemp())
    reader = TrajectoryReader(rd)

    windows = build_run_windows(
        reader, history_length=1, horizon=1, target="gradient_subspace",
        reps=["R1"], node_feat_dim=16,
    )
    G1 = torch.from_numpy(reader.layer_field("W1", "G", step=1))
    U1, _, _ = linalg.top_singular(G1, 4)
    got = torch.from_numpy(windows.Y[0, 0, : U1.numel()].reshape(U1.shape))

    assert torch.allclose(got, U1, atol=1e-6)
