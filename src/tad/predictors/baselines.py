"""Strong history-only baselines (Sections 7.2, 7.3, 11.1).

These are the comparison anchors: a learned predictor earns credit only by
beating the strongest applicable baseline (interpretation rules 7, 9), so these
must be tuned, not strawmen.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .base import Predictor, split_arrays
from ..utils import linalg

# matrix-valued gradient/update targets for which subspace / contraction
# baselines are meaningful; other targets fall back to persistence.
_MATRIX_TARGETS = {"future_gradient", "next_gradient", "future_update", "next_update"}


def _target_name(bundle) -> str:
    return str(bundle.get("target", "")) if bundle.get("target", None) is not None else ""


def _top_eigvecs(C: torch.Tensor, r: int) -> torch.Tensor:
    """Top-r eigenvectors of a symmetric PSD matrix (descending)."""
    C = 0.5 * (C + C.transpose(0, 1))
    evals, evecs = torch.linalg.eigh(C)
    return evecs[:, -r:].flip(1)


def _subspace_reconstruct(tgt_hist, shapes, rank, mode, beta=0.9, period=5) -> np.ndarray:
    """Predict a future gradient by projecting the last gradient onto an
    estimated slow subspace (Section 7.3 subspace baselines).

    mode='ema'      : exponentially weighted covariance eigenspace (online).
    mode='periodic' : subspace from SVD of the gradient at the last refresh step.
    """
    N, H, nodes, Td = tgt_hist.shape
    out = np.zeros((N, nodes, Td), dtype=np.float32)
    for ni in range(nodes):
        shape = tuple(int(s) for s in shapes[ni])
        flat = int(np.prod(shape))
        r = min(rank, shape[0], shape[1])
        ref = ((H - 1) // max(1, period)) * max(1, period) if mode == "periodic" else None
        for b in range(N):
            Glast = torch.from_numpy(tgt_hist[b, -1, ni, :flat].reshape(shape))
            if mode == "periodic":
                Gref = torch.from_numpy(tgt_hist[b, ref, ni, :flat].reshape(shape))
                U, _, V = linalg.top_singular(Gref, r)
            else:
                Cl = torch.zeros(shape[0], shape[0])
                Cr = torch.zeros(shape[1], shape[1])
                for t in range(H):
                    G = torch.from_numpy(tgt_hist[b, t, ni, :flat].reshape(shape))
                    Cl = beta * Cl + (1 - beta) * (G @ G.transpose(0, 1))
                    Cr = beta * Cr + (1 - beta) * (G.transpose(0, 1) @ G)
                U = _top_eigvecs(Cl, r)
                V = _top_eigvecs(Cr, r)
            C = U.transpose(0, 1) @ Glast @ V
            out[b, ni, :flat] = (U @ C @ V.transpose(0, 1)).reshape(-1).numpy()
    return out


class Persistence(Predictor):
    name = "persistence"

    def predict(self, bundle, split):
        _, tgt_hist, _ = split_arrays(bundle, split)
        return tgt_hist[:, -1]  # last observed target value


class ConstantVelocity(Predictor):
    name = "constant_velocity"

    def predict(self, bundle, split):
        _, tgt_hist, _ = split_arrays(bundle, split)
        if tgt_hist.shape[1] < 2:
            return tgt_hist[:, -1]
        vel = tgt_hist[:, -1] - tgt_hist[:, -2]
        return tgt_hist[:, -1] + vel


class TunedEMA(Predictor):
    """EMA over the target-space history; beta tuned on the val split."""

    name = "tuned_ema"

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.betas = list((cfg or {}).get("betas", [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]))
        self.beta = 0.9

    def _ema(self, tgt_hist, beta):
        out = tgt_hist[:, 0]
        for t in range(1, tgt_hist.shape[1]):
            out = beta * out + (1 - beta) * tgt_hist[:, t]
        return out

    def fit(self, bundle):
        feat, tgt_hist, Y = split_arrays(bundle, "val")
        if tgt_hist is None:
            feat, tgt_hist, Y = split_arrays(bundle, "train")
        best, best_err = self.betas[0], np.inf
        m = self.mask(bundle)[None]
        for b in self.betas:
            pred = self._ema(tgt_hist, b)
            err = float(np.sum(((pred - Y) * m) ** 2))
            if err < best_err:
                best, best_err = b, err
        self.beta = best

    def predict(self, bundle, split):
        _, tgt_hist, _ = split_arrays(bundle, split)
        return self._ema(tgt_hist, self.beta)


class Ridge(Predictor):
    """Ridge regression from flattened sketched features to the target."""

    name = "ridge"
    needs_training = True

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.alpha = float((cfg or {}).get("alpha", 1.0))
        self.W: Optional[np.ndarray] = None

    def fit(self, bundle):
        feat, _, Y = split_arrays(bundle, "train")
        X = feat.reshape(feat.shape[0], -1)
        X = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)
        Yf = Y.reshape(Y.shape[0], -1)
        A = X.T @ X + self.alpha * np.eye(X.shape[1], dtype=X.dtype)
        self.W = np.linalg.solve(A, X.T @ Yf)
        self._yshape = Y.shape[1:]

    def predict(self, bundle, split):
        feat, _, _ = split_arrays(bundle, split)
        X = feat.reshape(feat.shape[0], -1)
        X = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)
        return (X @ self.W).reshape(X.shape[0], *self._yshape)


class VectorAutoregression(Predictor):
    """Linear AR map from the flattened target-space history to the target."""

    name = "vector_autoregression"
    needs_training = True

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.alpha = float((cfg or {}).get("alpha", 1.0))

    def fit(self, bundle):
        _, tgt_hist, Y = split_arrays(bundle, "train")
        X = tgt_hist.reshape(tgt_hist.shape[0], -1)
        X = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)
        Yf = Y.reshape(Y.shape[0], -1)
        A = X.T @ X + self.alpha * np.eye(X.shape[1], dtype=X.dtype)
        self.W = np.linalg.solve(A, X.T @ Yf)
        self._yshape = Y.shape[1:]

    def predict(self, bundle, split):
        _, tgt_hist, _ = split_arrays(bundle, split)
        X = tgt_hist.reshape(tgt_hist.shape[0], -1)
        X = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)
        return (X @ self.W).reshape(X.shape[0], *self._yshape)


class DMDKoopman(Predictor):
    """DMD/Koopman-style linear latent transition fit on consecutive history.

    Learns A with S_{k+1} ~ A S_k over the history window, then rolls forward
    from the last state to the horizon.
    """

    name = "dmd_koopman"
    needs_training = True

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.alpha = float((cfg or {}).get("alpha", 1.0))
        self.A: Optional[np.ndarray] = None

    def fit(self, bundle):
        _, tgt_hist, _ = split_arrays(bundle, "train")
        N, H, nodes, Td = tgt_hist.shape
        if H < 2:
            self.A = None
            return
        X = tgt_hist[:, :-1].reshape(-1, nodes * Td)
        Xn = tgt_hist[:, 1:].reshape(-1, nodes * Td)
        G = X.T @ X + self.alpha * np.eye(X.shape[1], dtype=X.dtype)
        self.A = np.linalg.solve(G, X.T @ Xn)
        self._shape = (nodes, Td)
        # roll forward exactly h_idx base-grid steps (cadence-correct)
        self.rollout = max(1, int(bundle.get("h_idx", 1)))

    def predict(self, bundle, split):
        _, tgt_hist, _ = split_arrays(bundle, split)
        N, H, nodes, Td = tgt_hist.shape
        if self.A is None:
            return tgt_hist[:, -1]
        s = tgt_hist[:, -1].reshape(N, -1)
        for _ in range(self.rollout):
            s = s @ self.A
        return s.reshape(N, nodes, Td)


class OnlineSubspaceTracker(Predictor):
    """Online subspace tracker (Section 7.3): exponentially weighted covariance
    eigenspace of the gradient history (the fixed point of Oja's rule). Predicts
    the future gradient by projecting the last gradient onto the tracked left/
    right subspaces. Falls back to persistence for non-matrix targets.
    """

    name = "online_subspace_tracker"

    def __init__(self, cfg=None):
        super().__init__(cfg)
        cfg = cfg or {}
        self.rank = int(cfg.get("rank", 8))
        self.beta = float(cfg.get("ema_beta", 0.9))

    def predict(self, bundle, split):
        _, tgt_hist, _ = split_arrays(bundle, split)
        if _target_name(bundle) not in _MATRIX_TARGETS:
            return tgt_hist[:, -1]
        return _subspace_reconstruct(tgt_hist, bundle["node_shapes"], self.rank,
                                     mode="ema", beta=self.beta)


class PeriodicSVD(Predictor):
    """Periodic real-gradient SVD refresh (Section 7.3 / GaLore-style): the
    subspace is recomputed by SVD of the gradient at the last refresh step and
    held until the next refresh; predicts by projecting the last gradient onto
    it. Falls back to persistence for non-matrix targets.
    """

    name = "periodic_svd"

    def __init__(self, cfg=None):
        super().__init__(cfg)
        cfg = cfg or {}
        self.rank = int(cfg.get("rank", 8))
        self.period = int(cfg.get("refresh_period", 5))

    def predict(self, bundle, split):
        _, tgt_hist, _ = split_arrays(bundle, split)
        if _target_name(bundle) not in _MATRIX_TARGETS:
            return tgt_hist[:, -1]
        return _subspace_reconstruct(tgt_hist, bundle["node_shapes"], self.rank,
                                     mode="periodic", period=self.period)


class AnalyticContraction(Predictor):
    """Analytic deep-linear contraction baseline (Sections 7.4, 11.1(10), H4).

    For a 2-layer linear net, predicts the future end-to-end gradient G_M by EMA
    over its history, then forms the layer gradients via the exact contractions
    G_1 = W_2^T G_M, G_2 = G_M W_1^T using the anchor-step weights. A
    topology-aware learner is credited only for improvement beyond this.
    Requires the auxiliary tensors emitted for matrix-gradient targets;
    otherwise falls back to persistence.
    """

    name = "analytic_contraction"

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.beta = float((cfg or {}).get("ema_beta", 0.9))

    def predict(self, bundle, split):
        _, tgt_hist, _ = split_arrays(bundle, split)
        auxW = bundle.get(f"{split}_aux_W")
        auxGM = bundle.get(f"{split}_aux_GM_hist")
        names = list(bundle.get("layer_names", []))
        if (auxW is None or auxGM is None or len(names) != 2
                or _target_name(bundle) not in _MATRIX_TARGETS):
            return tgt_hist[:, -1]
        node_shapes = bundle["node_shapes"]
        Wshapes = bundle[f"{split}_aux_W_shapes"]
        GMshape = tuple(int(s) for s in bundle[f"{split}_aux_GM_shape"])
        gm_dim = int(np.prod(GMshape))
        N, nodes, Td = tgt_hist[:, -1].shape
        out = np.zeros((N, nodes, Td), dtype=np.float32)
        w1_shape = tuple(int(s) for s in Wshapes[0])
        w2_shape = tuple(int(s) for s in Wshapes[1])
        f1, f2 = int(np.prod(w1_shape)), int(np.prod(w2_shape))
        g1_flat = int(np.prod(tuple(int(s) for s in node_shapes[0])))
        g2_flat = int(np.prod(tuple(int(s) for s in node_shapes[1])))
        for b in range(N):
            gm = auxGM[b, 0, :gm_dim]
            for t in range(1, auxGM.shape[1]):
                gm = self.beta * gm + (1 - self.beta) * auxGM[b, t, :gm_dim]
            GM = gm.reshape(GMshape)
            W1 = auxW[b, 0, :f1].reshape(w1_shape)
            W2 = auxW[b, 1, :f2].reshape(w2_shape)
            out[b, 0, :g1_flat] = (W2.T @ GM).reshape(-1)
            out[b, 1, :g2_flat] = (GM @ W1.T).reshape(-1)
        return out


BASELINE_REGISTRY = {
    "persistence": Persistence,
    "constant_velocity": ConstantVelocity,
    "tuned_ema": TunedEMA,
    "ridge": Ridge,
    "vector_autoregression": VectorAutoregression,
    "dmd_koopman": DMDKoopman,
    "online_subspace_tracker": OnlineSubspaceTracker,
    "periodic_svd": PeriodicSVD,
    "analytic_contraction": AnalyticContraction,
}
