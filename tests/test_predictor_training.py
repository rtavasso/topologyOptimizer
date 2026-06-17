import numpy as np
import torch
import torch.nn as nn

from tad.predictors._torch_util import train_module, predict_module


class _LastStepLinear(nn.Module):
    def __init__(self, feat_dim, tgt_dim):
        super().__init__()
        self.head = nn.Linear(feat_dim, tgt_dim)

    def forward(self, feat, tgt_hist):
        return self.head(feat[:, -1])


def test_torch_predictor_training_normalizes_large_scale_targets():
    rng = np.random.default_rng(0)
    n, h, nodes, feat_dim, tgt_dim = 96, 3, 2, 4, 3
    feat = (1000.0 * rng.standard_normal((n, h, nodes, feat_dim))).astype(np.float32)
    W = rng.standard_normal((feat_dim, tgt_dim)).astype(np.float32)
    Y = (feat[:, -1] @ W + 1.0e6).astype(np.float32)
    tgt_hist = np.repeat(Y[:, None], h, axis=1)
    mask = np.ones((nodes, tgt_dim), dtype=np.float32)

    bundle = {
        "train_feat": feat[:64], "train_tgt_hist": tgt_hist[:64], "train_Y": Y[:64],
        "val_feat": feat[64:80], "val_tgt_hist": tgt_hist[64:80], "val_Y": Y[64:80],
        "test_feat": feat[80:], "test_tgt_hist": tgt_hist[80:], "test_Y": Y[80:],
        "mask": mask,
    }
    module = _LastStepLinear(feat_dim, tgt_dim)
    fwd = lambda m, f, t: m(f, t)

    train_module(module, bundle, fwd, epochs=80, lr=3e-3, batch_size=32)
    pred = predict_module(module, bundle, "test", fwd)

    assert np.isfinite(pred).all()
    assert np.mean((pred - Y[80:]) ** 2) < np.var(Y[80:])
