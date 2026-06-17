"""Common predictor interface and split accessors.

A predictor maps a windowed dataset bundle to per-node predictions in the target
space, shaped (N, num_nodes, Tdim). History-only baselines read ``tgt_hist``;
learned models read ``feat_hist`` (sketched representation features). All
predictions are compared under the per-node ``mask``.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def split_arrays(bundle: dict, split: str):
    feat = bundle.get(f"{split}_feat")
    tgt_hist = bundle.get(f"{split}_tgt_hist")
    Y = bundle.get(f"{split}_Y")
    return feat, tgt_hist, Y


class Predictor:
    name = "base"
    needs_training = False

    def __init__(self, cfg=None):
        self.cfg = cfg or {}

    def fit(self, bundle: dict) -> None:
        pass

    def predict(self, bundle: dict, split: str) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def mask(bundle: dict) -> np.ndarray:
        return bundle["mask"]
