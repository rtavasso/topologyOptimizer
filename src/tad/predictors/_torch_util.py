"""Shared training harness for learned predictors."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .base import split_arrays
from ..utils.profiling import ComputeLedger


def masked_mse(pred, target, mask):
    m = mask.unsqueeze(0)
    diff = (pred - target) * m
    denom = m.sum() * pred.shape[0] + 1e-8
    return (diff ** 2).sum() / denom


def train_module(
    module: nn.Module,
    bundle: dict,
    forward_fn,
    epochs: int = 60,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: str = "cpu",
    weight_decay: float = 1e-5,
    ledger: Optional[ComputeLedger] = None,
):
    """Generic masked-MSE training loop with early stopping on val.

    ``forward_fn(module, feat, tgt_hist)`` returns predictions (B, nodes, Tdim).
    """
    feat, tgt_hist, Y = split_arrays(bundle, "train")
    mask = torch.from_numpy(bundle["mask"]).float().to(device)
    feat_t = torch.from_numpy(feat).float().to(device)
    tgt_t = torch.from_numpy(tgt_hist).float().to(device)
    Y_t = torch.from_numpy(Y).float().to(device)

    vfeat, vtgt, vY = split_arrays(bundle, "val")
    has_val = vfeat is not None and len(vfeat) > 0
    if has_val:
        vfeat_t = torch.from_numpy(vfeat).float().to(device)
        vtgt_t = torch.from_numpy(vtgt).float().to(device)
        vY_t = torch.from_numpy(vY).float().to(device)

    module = module.to(device)
    opt = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=weight_decay)
    n = feat_t.shape[0]
    best_state, best_val = None, np.inf

    for ep in range(epochs):
        module.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pred = forward_fn(module, feat_t[idx], tgt_t[idx])
            loss = masked_mse(pred, Y_t[idx], mask)
            loss.backward()
            opt.step()
            if ledger is not None:
                ledger.add_flops("predictor_train", sum(p.numel() for p in module.parameters()) * 6 * len(idx))
        if has_val:
            module.eval()
            with torch.no_grad():
                vpred = forward_fn(module, vfeat_t, vtgt_t)
                vloss = float(masked_mse(vpred, vY_t, mask).item())
            if vloss < best_val:
                best_val = vloss
                best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
    if best_state is not None:
        module.load_state_dict(best_state)
    return module


def predict_module(module, bundle, split, forward_fn, device="cpu") -> np.ndarray:
    feat, tgt_hist, _ = split_arrays(bundle, split)
    module.eval()
    with torch.no_grad():
        feat_t = torch.from_numpy(feat).float().to(device)
        tgt_t = torch.from_numpy(tgt_hist).float().to(device)
        pred = forward_fn(module, feat_t, tgt_t)
    return pred.cpu().numpy()
