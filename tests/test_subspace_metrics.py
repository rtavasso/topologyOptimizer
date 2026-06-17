"""SVD sign/subspace invariance (spec Section 20.5)."""
import torch

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
