"""Linear-algebra helpers: SVD, subspace metrics, ranks, sign alignment.

All functions accept and return torch tensors and are written to be
sign/rotation invariant where the spec requires it (Section 20.5).
"""
from __future__ import annotations

from typing import Tuple

import torch

EPS = 1e-12


def safe_svd(A: torch.Tensor, full_matrices: bool = False):
    """SVD that sanitizes non-finite inputs and retries in double precision."""
    if not torch.isfinite(A).all():
        A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        return torch.linalg.svd(A, full_matrices=full_matrices)
    except Exception:
        A64 = torch.nan_to_num(A.cpu().double())
        U, S, Vh = torch.linalg.svd(A64, full_matrices=full_matrices)
        return U.to(A.dtype), S.to(A.dtype), Vh.to(A.dtype)


def top_singular(A: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return top-``rank`` (U, S, V) with V (not Vh).

    U: (m, r), S: (r,), V: (n, r).
    """
    U, S, Vh = safe_svd(A, full_matrices=False)
    r = min(rank, S.shape[0])
    U = U[:, :r]
    S = S[:r]
    V = Vh[:r, :].transpose(-2, -1)
    return sign_align(U, V, S)


def sign_align(U: torch.Tensor, V: torch.Tensor, S: torch.Tensor):
    """Fix SVD sign ambiguity deterministically.

    Convention: make the largest-magnitude entry of each left singular vector
    positive, and flip the matching right singular vector to preserve the sign
    of the reconstruction. This makes logged singular vectors comparable across
    steps and removes spurious sign flips from temporal metrics.
    """
    idx = U.abs().argmax(dim=0)
    signs = torch.sign(U[idx, torch.arange(U.shape[1], device=U.device)])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    U = U * signs.unsqueeze(0)
    V = V * signs.unsqueeze(0)
    return U, S, V


def projector(U: torch.Tensor) -> torch.Tensor:
    """Orthogonal projector U U^T (sign/rotation invariant within the subspace)."""
    return U @ U.transpose(-2, -1)


def subspace_overlap(U1: torch.Tensor, U2: torch.Tensor) -> float:
    """Normalized overlap (1/r) ||U1^T U2||_F^2 in [0, 1] (Section 13.3)."""
    r = U1.shape[1]
    if r == 0:
        return 0.0
    M = U1.transpose(-2, -1) @ U2
    return float((M.pow(2).sum() / r).item())


def projection_distance(U1: torch.Tensor, U2: torch.Tensor) -> float:
    """||U1 U1^T - U2 U2^T||_F (Section 12.7), invariant to bases/sign."""
    return float(torch.linalg.norm(projector(U1) - projector(U2)).item())


def energy_capture(U: torch.Tensor, G: torch.Tensor, V: torch.Tensor) -> float:
    """Fraction of energy of G captured by the (U, V) subspace (Section 13.4).

    R = ||U^T G V||_F^2 / ||G||_F^2.
    """
    num = torch.linalg.norm(U.transpose(-2, -1) @ G @ V).pow(2)
    den = torch.linalg.norm(G).pow(2) + EPS
    return float((num / den).item())


def effective_rank(A: torch.Tensor, energy: bool = True) -> float:
    """Spectral-entropy effective rank.

    Default convention (Section 24.17): normalized singular-value *energy*
    p_i = sigma_i^2 / sum_j sigma_j^2.
    """
    S = safe_svd(A)[1]
    S = S[S > EPS]
    if S.numel() == 0:
        return 0.0
    w = S.pow(2) if energy else S
    p = w / w.sum()
    H = -(p * (p + EPS).log()).sum()
    return float(H.exp().item())


def stable_rank(A: torch.Tensor) -> float:
    """||A||_F^2 / ||A||_2^2 (Section 13.5)."""
    fro2 = torch.linalg.norm(A).pow(2)
    spec2 = safe_svd(A)[1].max().pow(2) + EPS
    return float((fro2 / spec2).item())


def condition_number(A: torch.Tensor) -> float:
    S = safe_svd(A)[1]
    S = S[S > EPS]
    if S.numel() == 0:
        return float("inf")
    return float((S.max() / S.min()).item())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Flattened cosine similarity."""
    a = a.reshape(-1)
    b = b.reshape(-1)
    denom = a.norm() * b.norm() + EPS
    return float((a @ b / denom).item())


def matrix_inv_sqrt(C: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Regularized inverse square root (C + eps I)^{-1/2} (Section 10 R6)."""
    n = C.shape[-1]
    C = 0.5 * (C + C.transpose(-2, -1))
    evals, evecs = torch.linalg.eigh(C + eps * torch.eye(n, device=C.device, dtype=C.dtype))
    evals = evals.clamp_min(eps)
    return (evecs * evals.rsqrt().unsqueeze(-2)) @ evecs.transpose(-2, -1)


def matrix_sqrt(C: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    n = C.shape[-1]
    C = 0.5 * (C + C.transpose(-2, -1))
    evals, evecs = torch.linalg.eigh(C + eps * torch.eye(n, device=C.device, dtype=C.dtype))
    evals = evals.clamp_min(0.0)
    return (evecs * evals.sqrt().unsqueeze(-2)) @ evecs.transpose(-2, -1)


def normalized_mse(pred: torch.Tensor, target: torch.Tensor, eps: float = EPS) -> float:
    """||pred - target||_F^2 / (||target||_F^2 + eps)."""
    num = torch.linalg.norm((pred - target).reshape(-1)).pow(2)
    den = torch.linalg.norm(target.reshape(-1)).pow(2) + eps
    return float((num / den).item())
