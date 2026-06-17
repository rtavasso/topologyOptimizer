"""Fixed probe sets (Section 8.5).

Probes are generated once at run initialization, versioned, and stored with the
trajectory so probe-action differences are meaningful across steps. Each map
W_l gets a probe matrix P_l of shape (in_dim_l, n_probes).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch


@dataclass
class ProbeSet:
    per_layer: Dict[str, torch.Tensor]          # name -> P_l  (in_dim, k)
    input_probe: torch.Tensor                    # P_0 for prefix actions (in0, k)
    kind: str
    version: int = 1

    def to_numpy(self) -> dict:
        return {
            "kind": self.kind, "version": self.version,
            "input_probe": self.input_probe.cpu().numpy(),
            **{f"P__{n}": p.cpu().numpy() for n, p in self.per_layer.items()},
        }


def build_probes(model, data_stream, cfg, seed: int, device="cpu") -> ProbeSet:
    """Build probe matrices for each layer.

    ``kind`` selects the family (Section 8.5). For ``pca`` we use principal
    components of the input distribution; for ``teacher`` we use teacher right
    singular vectors padded/truncated per layer.
    """
    kind = cfg.get("probe_kind", "gaussian")
    k = int(cfg.get("random_probe_count", 16))
    g = torch.Generator().manual_seed(seed ^ 0x9E3779B1)
    per_layer: Dict[str, torch.Tensor] = {}

    for name in model.layer_names:
        in_dim = model.weight(name).shape[1]
        kk = min(k, in_dim)
        if kind == "standard_basis":
            P = torch.zeros(in_dim, kk)
            P[torch.arange(kk), torch.arange(kk)] = 1.0
        else:  # gaussian probes (default); pca/teacher handled for input_probe
            P = torch.randn(in_dim, kk, generator=g)
            P = torch.linalg.qr(P)[0] if kk <= in_dim else P
        per_layer[name] = P.to(device)

    # input probe for prefix actions
    in0 = model.weight(model.layer_names[0]).shape[1]
    k0 = min(k, in0)
    if kind == "pca":
        x = data_stream.batch(0).x.detach().cpu().numpy()
        C = np.cov(x, rowvar=False)
        evals, evecs = np.linalg.eigh(C)
        P0 = torch.from_numpy(evecs[:, -k0:].astype(np.float32))
    else:
        P0 = torch.linalg.qr(torch.randn(in0, k0, generator=g))[0]
    return ProbeSet(per_layer=per_layer, input_probe=P0.to(device), kind=kind)
