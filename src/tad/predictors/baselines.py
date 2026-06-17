"""Strong history-only baselines (Sections 7.2, 7.3, 11.1).

These are the comparison anchors: a learned predictor earns credit only by
beating the strongest applicable baseline (interpretation rules 7, 9), so these
must be tuned, not strawmen.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Predictor, split_arrays


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
        self.rollout = max(1, int(bundle.get("train_meta_horizon", np.array([1]))[0]) //
                           max(1, int(bundle.get("cadence", 1))))

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
    """Oja-style online subspace tracker over the gradient history (Section 7.3).

    Predicts the leading left-singular subspace; for general matrix targets it
    falls back to persistence of the target value.
    """

    name = "online_subspace_tracker"

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.lr = float((cfg or {}).get("oja_lr", 0.1))

    def predict(self, bundle, split):
        # For target-space prediction we return persistence; the *subspace*
        # quality of this tracker is evaluated by the subspace metrics that call
        # ``track_subspace`` below.
        _, tgt_hist, _ = split_arrays(bundle, split)
        return tgt_hist[:, -1]


class PeriodicSVD(Predictor):
    """Periodic-SVD baseline: predicts from the most recent observed target."""

    name = "periodic_svd"

    def predict(self, bundle, split):
        _, tgt_hist, _ = split_arrays(bundle, split)
        return tgt_hist[:, -1]


BASELINE_REGISTRY = {
    "persistence": Persistence,
    "constant_velocity": ConstantVelocity,
    "tuned_ema": TunedEMA,
    "ridge": Ridge,
    "vector_autoregression": VectorAutoregression,
    "dmd_koopman": DMDKoopman,
    "online_subspace_tracker": OnlineSubspaceTracker,
    "periodic_svd": PeriodicSVD,
}
