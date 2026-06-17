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


def _normalization_stats(feat_t, tgt_t, Y_t, mask):
    feat_mean = feat_t.mean(dim=(0, 1, 2), keepdim=True)
    feat_std = feat_t.std(dim=(0, 1, 2), keepdim=True).clamp_min(1e-6)

    m = mask.unsqueeze(0)
    denom = (mask.unsqueeze(0) * Y_t.shape[0]).clamp_min(1.0)
    y_mean = (Y_t * m).sum(dim=0, keepdim=True) / denom
    y_var = (((Y_t - y_mean) * m) ** 2).sum(dim=0, keepdim=True) / denom
    y_std = y_var.sqrt().clamp_min(1e-6)
    y_mean = torch.where(mask.unsqueeze(0) > 0, y_mean, torch.zeros_like(y_mean))
    y_std = torch.where(mask.unsqueeze(0) > 0, y_std, torch.ones_like(y_std))
    return feat_mean, feat_std, y_mean, y_std


def _attach_stats(module, feat_mean, feat_std, y_mean, y_std):
    module._tad_norm = {
        "feat_mean": feat_mean.detach().cpu(),
        "feat_std": feat_std.detach().cpu(),
        "y_mean": y_mean.detach().cpu(),
        "y_std": y_std.detach().cpu(),
    }


def _stats_on(module, device):
    st = getattr(module, "_tad_norm", None)
    if st is None:
        return None
    return {k: v.to(device) for k, v in st.items()}


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
    feat_mean, feat_std, y_mean, y_std = _normalization_stats(feat_t, tgt_t, Y_t, mask)
    _attach_stats(module, feat_mean, feat_std, y_mean, y_std)
    feat_t = (feat_t - feat_mean) / feat_std
    tgt_t = (tgt_t - y_mean.unsqueeze(1)) / y_std.unsqueeze(1)
    Y_t = (Y_t - y_mean) / y_std

    vfeat, vtgt, vY = split_arrays(bundle, "val")
    has_val = vfeat is not None and len(vfeat) > 0
    if has_val:
        vfeat_t = (torch.from_numpy(vfeat).float().to(device) - feat_mean) / feat_std
        vtgt_t = (torch.from_numpy(vtgt).float().to(device) - y_mean.unsqueeze(1)) / y_std.unsqueeze(1)
        vY_t = (torch.from_numpy(vY).float().to(device) - y_mean) / y_std

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
        st = _stats_on(module, device)
        if st is not None:
            feat_t = (feat_t - st["feat_mean"]) / st["feat_std"]
            tgt_t = (tgt_t - st["y_mean"].unsqueeze(1)) / st["y_std"].unsqueeze(1)
        pred = forward_fn(module, feat_t, tgt_t)
        if st is not None:
            pred = pred * st["y_std"] + st["y_mean"]
    return pred.cpu().numpy()
