"""Per-step logging hook (Sections 8.2-8.4).

Computes per-layer functional fields from captured activations/errors, network
topology fields (products, balance, ranks), and writes them through the
TrajectoryWriter on the appropriate cadence grid. Pure observation: it only
reads ``.detach()``ed tensors and never changes training.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from ..topology import products, invariants
from ..utils import linalg
from .optimizers import optimizer_state_snapshot
from ..logging import spectral


def _cov(X: torch.Tensor) -> torch.Tensor:
    Xc = X - X.mean(0, keepdim=True)
    return (Xc.transpose(0, 1) @ Xc) / max(1, X.shape[0])


class LoggingHook:
    def __init__(self, model, writer, probes, optimizer, cadences,
                 svd_rank: int = 16, is_linear: bool = True):
        self.model = model
        self.writer = writer
        self.probes = probes
        self.optimizer = optimizer
        self.cad = cadences
        self.svd_rank = svd_rank
        self.is_linear = is_linear
        self.param_to_name = {id(p): n for n, p in model.named_parameters()}
        # input probe P0 for prefix actions
        self.P0 = probes.input_probe if probes is not None else None

    def on_step(self, step: int, pre_weights: Dict[str, torch.Tensor],
                global_scalars: dict) -> None:
        do_full = step % self.cad.full_tensor_every == 0
        do_svd = step % self.cad.svd_every == 0
        opt_snap = optimizer_state_snapshot(self.optimizer, self.param_to_name)

        names = self.model.layer_names
        # Canonical state at step t uses the PRE-step weights (the point at which
        # G_t was computed), so W_t, G_t, and the contractions are consistent;
        # DeltaW_t is the applied update post - pre.
        weights = [pre_weights[n] for n in names]

        # -- per layer -------------------------------------------------------
        prefix = products.prefix_products(weights)
        for i, name in enumerate(names):
            lin = self.model.linear(name)
            W = pre_weights[name]
            G = lin.weight.grad.detach() if lin.weight.grad is not None else None
            dW = (lin.weight.detach() - pre_weights[name]) if name in pre_weights else None
            cap = self.model.captures.get(name)

            # scalar fields (full grid)
            if do_full:
                self.writer.log_scalars(step, {
                    f"{name}__weight_frobenius_norm": float(torch.linalg.norm(W).item()),
                    f"{name}__gradient_frobenius_norm": float(torch.linalg.norm(G).item()) if G is not None else None,
                    f"{name}__update_frobenius_norm": float(torch.linalg.norm(dW).item()) if dW is not None else None,
                    f"{name}__gradient_momentum_cosine": self._grad_mom_cos(name, G, opt_snap),
                    f"{name}__effective_rank_W": linalg.effective_rank(W),
                    f"{name}__stable_rank_W": linalg.stable_rank(W),
                    f"{name}__condition_number_W": linalg.condition_number(W),
                })

            if do_full:
                self.writer.log_layer_full(step, name, "W", W)
                self.writer.log_layer_full(step, name, "G", G)
                self.writer.log_layer_full(step, name, "DeltaW", dW)
                entry = opt_snap.get(name, {})
                self.writer.log_layer_full(step, name, "momentum", entry.get("momentum"))
                self.writer.log_layer_full(step, name, "second_moment", entry.get("second_moment"))
                # gram + probes
                self.writer.log_layer_full(step, name, "gram_in", W.transpose(0, 1) @ W)
                self.writer.log_layer_full(step, name, "gram_out", W @ W.transpose(0, 1))
                P = self.probes.per_layer[name]
                self.writer.log_layer_full(step, name, "probe_WP", W @ P)
                if G is not None:
                    self.writer.log_layer_full(step, name, "probe_GP", G @ P)
                self.writer.log_layer_full(step, name, "probe_prefix", prefix[i] @ self.P0)
                # functional covariances from captures
                if cap is not None and cap.pre_input is not None:
                    self.writer.log_layer_full(step, name, "activation_mean", cap.pre_input.mean(0))
                    self.writer.log_layer_full(step, name, "activation_covariance", _cov(cap.pre_input))
                if cap is not None and cap.pre_output is not None:
                    self.writer.log_layer_full(step, name, "output_covariance", _cov(cap.pre_output))
                if cap is not None and cap.act_mask is not None:
                    self.writer.log_layer_full(step, name, "activation_mask", cap.act_mask)
                if cap is not None and cap.out_grad is not None:
                    self.writer.log_layer_full(step, name, "backprop_error_mean", cap.out_grad.mean(0))
                    self.writer.log_layer_full(step, name, "backprop_error_covariance", _cov(cap.out_grad))
                    if cap.pre_input is not None:
                        cc = (cap.out_grad.transpose(0, 1) @ cap.pre_input) / max(1, cap.pre_input.shape[0])
                        self.writer.log_layer_full(step, name, "cross_covariance_error_input", cc)

            if do_svd:
                for f, t in spectral.spectral_fields(name, W, G, dW, self.svd_rank).items():
                    self.writer.log_layer_svd(step, name, f, t)

        # -- network ---------------------------------------------------------
        if do_full:
            M = prefix[-1]
            self.writer.log_network_full(step, "end_to_end_map", M)
            suffix = products.suffix_products(weights)
            for name, P in zip(names, prefix):
                self.writer.log_network_full(step, f"prefix_product__{name}", P)
            for name, S in zip(names, suffix):
                self.writer.log_network_full(step, f"suffix_product__{name}", S)
            if self.is_linear:
                Gm = self._end_to_end_gradient()
                if Gm is not None:
                    self.writer.log_network_full(step, "end_to_end_gradient", Gm)
            J = self._effective_local_map(weights)
            if J is not None:
                self.writer.log_network_full(step, "effective_local_map", J)
                if self.P0 is not None:
                    self.writer.log_network_full(step, "probe_effective_local_map", J @ self.P0)
            balance = invariants.all_balance_errors(weights)
            conds = [linalg.condition_number(W) for W in weights]
            stable = [linalg.stable_rank(W) for W in weights]
            effective = [linalg.effective_rank(W) for W in weights]
            self.writer.log_network_full(step, "balance_errors", torch.tensor(balance, dtype=M.dtype, device=M.device))
            self.writer.log_network_full(step, "condition_numbers", torch.tensor(conds, dtype=M.dtype, device=M.device))
            self.writer.log_network_full(step, "stable_ranks", torch.tensor(stable, dtype=M.dtype, device=M.device))
            self.writer.log_network_full(step, "effective_ranks", torch.tensor(effective, dtype=M.dtype, device=M.device))
            self.writer.log_scalars(step, {
                "balance_error_mean": float(sum([b for b in balance if b == b]) / max(1, len(balance))) if balance else None,
                "end_to_end_effective_rank": linalg.effective_rank(M),
                "end_to_end_stable_rank": linalg.stable_rank(M),
            })

    def _grad_mom_cos(self, name, G, opt_snap) -> Optional[float]:
        if G is None:
            return None
        mom = opt_snap.get(name, {}).get("momentum")
        if mom is None:
            return None
        return linalg.cosine(G, mom)

    def _end_to_end_gradient(self) -> Optional[torch.Tensor]:
        """G_M = (last-layer out_grad)^T @ (first-layer input) for linear stacks."""
        names = self.model.layer_names
        first = self.model.captures.get(names[0])
        last = self.model.captures.get(names[-1])
        if first is None or last is None:
            return None
        if first.pre_input is None or last.out_grad is None:
            return None
        return last.out_grad.transpose(0, 1) @ first.pre_input

    def _effective_local_map(self, weights: List[torch.Tensor]) -> Optional[torch.Tensor]:
        """Probe-averaged nonlinear local Jacobian from captured activation masks."""
        if not any(getattr(self.model.captures.get(n), "act_mask", None) is not None for n in self.model.layer_names):
            return None
        J = None
        for i, (name, W) in enumerate(zip(self.model.layer_names, weights)):
            J = W if J is None else W @ J
            cap = self.model.captures.get(name)
            mask = getattr(cap, "act_mask", None) if cap is not None else None
            if mask is not None and i < len(weights) - 1:
                J = mask.to(device=J.device, dtype=J.dtype).unsqueeze(1) * J
        return J
