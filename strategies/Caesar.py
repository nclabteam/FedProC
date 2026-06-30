# -*- coding: utf-8 -*-
"""Caesar - A Low-deviation Compression Approach for Efficient Federated Learning.

Paper: https://arxiv.org/abs/2412.19989

Three-component framework (batch-size optimization omitted -- simulation only):

Download (server -> client):
  Staleness-aware compression ratio theta_d_i = (1 - delta_i/t) * theta_d_max.
  Top-(1-theta_d) elements sent at full precision (index + float32 value);
  bottom-theta_d elements sent as 1-bit sign + (avg_abs, max_abs) per tensor.
  Client recovers compressed elements from its previous local model where
  sign matches and abs <= max_abs; otherwise uses avg_abs * sign.
  Wire size per tensor ≈ n_full*5 + n_comp*1 bytes (float32 + int64 idx for
  full; 1-bit sign packed into int8 for comp; two float32 scalars per tensor).

Upload (client -> server):
  Importance C_i = train_samples / max_samples (TSF adaptation: KL-divergence
  term from Eq. 3 dropped as no class labels exist).
  Rank-based upload ratio over all N clients (pre-computed, fixed):
    theta_u_i = theta_u_min + (theta_u_max - theta_u_min) / N * rank(C_i)
  Client sparsifies gradient (top-K by magnitude, keeping 1-theta_u fraction)
  and transmits as COO sparse format (index + value per retained element).
  Wire size ≈ n_keep * 12 bytes (int64 index + float32 value).

_caesar_final_params is carried in the package for simulation bookkeeping
(stateless clients cannot persist state); it is not part of the paper's wire
protocol and is excluded from uplink measurement.

Aggregation: w^{t+1} = w^t - (1/|N^t|) * sum(compressed_gradient_i).
"""
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch

from .tFL import tFL, tFL_Client

_BYTES_PER_FLOAT32 = 4
_BYTES_PER_INT64 = 8
_BYTES_PER_MB = 1024 ** 2


def _compressed_downlink_bytes(compressed: Optional[Dict[str, Any]]) -> float:
    """Exact wire size of a Caesar compressed-model dict in bytes."""
    if compressed is None:
        return 0.0
    total = 0.0
    for cdata in compressed.values():
        n_full = len(cdata["full_idx"])
        n_comp = len(cdata["comp_idx"])
        # full elements: float32 value + int64 index
        total += n_full * (_BYTES_PER_FLOAT32 + _BYTES_PER_INT64)
        # compressed elements: 1-bit sign (packed into int8) + 2 float32 scalars per tensor
        total += n_comp * 1 + 2 * _BYTES_PER_FLOAT32
    return total


def _sparse_gradient_bytes(grad: OrderedDict) -> float:
    """COO wire size for a sparsified gradient (non-zero elements only)."""
    total = 0.0
    for g in grad.values():
        n_keep = int(g.view(-1).count_nonzero().item())
        # int64 index + float32 value per retained element
        total += n_keep * (_BYTES_PER_INT64 + _BYTES_PER_FLOAT32)
    return total


class Caesar(tFL):
    """Caesar server - staleness-aware download + importance-ranked gradient upload."""

    optional = {
        "theta_d_max": 0.5,
        "theta_u_min": 0.1,
        "theta_u_max": 0.5,
    }

    def __init__(self, configs, times):
        super().__init__(configs, times)
        self._caesar_last_round: Dict[int, int] = {}
        self._caesar_importance: Dict[int, float] = {}
        self._caesar_upload_ratio: Dict[int, float] = {}
        self._caesar_prev_params: Dict[int, Optional[OrderedDict]] = {}
        self._caesar_ratio_iter: int = -1
        self._caesar_downlink_mb: Dict[int, float] = {}  # true wire size, bypasses _dispatch overwrite

    def _caesar_init_importance(self, packages: Dict[int, Any]) -> None:
        for cid, pkg in packages.items():
            self._caesar_importance[cid] = float(pkg["score"])
        max_s = max(self._caesar_importance.values()) or 1.0
        for cid in self._caesar_importance:
            self._caesar_importance[cid] /= max_s

    def _caesar_update_upload_ratios(self, selected: List[int]) -> None:
        n = self.num_clients
        ranked = sorted(selected, key=lambda c: self._caesar_importance.get(c, 0.0), reverse=True)
        for rank, cid in enumerate(ranked, start=1):
            self._caesar_upload_ratio[cid] = (
                self.theta_u_min + (self.theta_u_max - self.theta_u_min) / n * rank
            )

    def _caesar_compress(self, theta_d: float) -> Optional[Dict[str, Any]]:
        if theta_d <= 0.0:
            return None
        compressed = {}
        for name, param in self.model.named_parameters():
            flat = param.data.view(-1).float().cpu()
            n = flat.numel()
            n_full = max(1, int(n * (1.0 - theta_d)))
            _, order = torch.sort(flat.abs(), descending=True)
            full_idx = order[:n_full]
            comp_idx = order[n_full:]
            comp_vals = flat[comp_idx]
            avg_abs = float(comp_vals.abs().mean()) if len(comp_idx) > 0 else 0.0
            max_abs = float(comp_vals.abs().max()) if len(comp_idx) > 0 else 0.0
            compressed[name] = {
                "shape": tuple(param.shape),
                "n": n,
                "full_idx": full_idx,
                "full_vals": flat[full_idx].clone(),
                "comp_idx": comp_idx,
                "comp_signs": torch.sign(comp_vals).clone(),
                "avg_abs": avg_abs,
                "max_abs": max_abs,
            }
        return compressed

    def package(self, client_id: int) -> Dict[str, Any]:
        # Compute upload ratios once per round for all selected clients
        if self._caesar_ratio_iter != self.current_iter and self._caesar_importance:
            self._caesar_update_upload_ratios(self.selected_clients)
            self._caesar_ratio_iter = self.current_iter

        t = max(self.current_iter, 1)
        delta = t - self._caesar_last_round.get(client_id, 0)
        theta_d = (1.0 - delta / t) * self.theta_d_max
        theta_u = self._caesar_upload_ratio.get(client_id, self.theta_u_min)

        pkg = super().package(client_id)
        compressed = self._caesar_compress(theta_d)
        pkg["_caesar_compressed"] = compressed
        pkg["_caesar_prev_params"] = self._caesar_prev_params.get(client_id)
        pkg["_caesar_theta_u"] = theta_u

        # Record true downlink wire size (compressed dict when theta_d>0, full model otherwise)
        if compressed is not None:
            self._caesar_downlink_mb[client_id] = (
                _compressed_downlink_bytes(compressed) / _BYTES_PER_MB
            )
        else:
            self._caesar_downlink_mb[client_id] = self.get_size(self.public_model_params)
        return pkg

    def _compute_send_mb(self, packages) -> tuple:
        # Uplink: COO sparse gradient size per client (excludes _caesar_final_params)
        uplink = {}
        for cid, pkg in packages.items():
            grad = pkg.get("_caesar_gradient", {})
            uplink[cid] = _sparse_gradient_bytes(grad) / _BYTES_PER_MB

        # Downlink: sum of per-client true wire sizes (set in package() before _dispatch overwrites)
        downlink = sum(self._caesar_downlink_mb.get(cid, 0.0) for cid in self.selected_clients)
        return uplink, downlink

    def aggregate_client_updates(self, packages: "OrderedDict[int, dict]") -> None:
        # Bootstrap importance on first round
        if not self._caesar_importance:
            self._caesar_init_importance(packages)
            self._caesar_update_upload_ratios(list(packages.keys()))
            self._caesar_ratio_iter = self.current_iter

        n_participants = len(packages)
        w = 1.0 / n_participants

        avg_grad: Dict[str, torch.Tensor] = {}
        for pkg in packages.values():
            for name, g in pkg["_caesar_gradient"].items():
                if name not in avg_grad:
                    avg_grad[name] = g.float() * w
                else:
                    avg_grad[name] += g.float() * w

        # w^{t+1} = w^t - avg_gradient
        new_params = OrderedDict()
        for name in self.public_model_params:
            delta = avg_grad.get(name, torch.zeros_like(self.public_model_params[name]))
            new_params[name] = (self.public_model_params[name].float() - delta).to(
                self.public_model_params[name].dtype
            )
        self._commit_global(new_params)

        # Store client final params + update participation records
        for cid, pkg in packages.items():
            self._caesar_prev_params[cid] = pkg["_caesar_final_params"]
            self._caesar_last_round[cid] = self.current_iter


class Caesar_Client(tFL_Client):
    """Caesar client - model recovery + gradient compression."""

    def set_parameters(self, package: Dict[str, Any]) -> None:
        super().set_parameters(package)
        self._caesar_theta_u = package["_caesar_theta_u"]
        compressed = package["_caesar_compressed"]
        if compressed is not None:
            self._caesar_recover(compressed, package["_caesar_prev_params"])

    def _caesar_recover(
        self,
        compressed: Dict[str, Any],
        prev_params: Optional[OrderedDict],
    ) -> None:
        """Overwrite model state with recovered approximation of global model."""
        state = {}
        for name, cdata in compressed.items():
            flat = torch.zeros(cdata["n"])
            flat[cdata["full_idx"]] = cdata["full_vals"]
            comp_idx = cdata["comp_idx"]
            if len(comp_idx) > 0:
                signs = cdata["comp_signs"].float()
                if prev_params is not None and name in prev_params:
                    prev_comp = prev_params[name].view(-1).float()[comp_idx]
                    sign_ok = torch.sign(prev_comp) == signs
                    abs_ok = prev_comp.abs() <= cdata["max_abs"]
                    recovered = torch.where(sign_ok & abs_ok, prev_comp, signs * cdata["avg_abs"])
                else:
                    recovered = signs * cdata["avg_abs"]
                flat[comp_idx] = recovered
            state[name] = flat.view(cdata["shape"])
        self.model.load_state_dict(state, strict=False)

    def fit(self) -> None:
        from .base import SharedMethods
        SharedMethods._set_worker_seed(self._loader_seed("train"))

        # Snapshot starting params to compute gradient after training
        init_params = OrderedDict(
            (n, p.data.cpu().clone()) for n, p in self.model.named_parameters()
        )

        loader = self.load_train_data()
        offload = self.efficiency == "low"
        for _ in range(self.epochs):
            self.train_one_epoch(
                model=self.model,
                dataloader=loader,
                optimizer=self.optimizer,
                criterion=self.loss,
                scheduler=self.scheduler,
                device=self.device,
                offload_after=offload,
            )
        if self.efficiency == "med":
            self.model.to("cpu")

        # gradient = init - final  (server applies: w -= avg_grad)
        self._caesar_gradient = OrderedDict(
            (n, init_params[n] - p.data.cpu()) for n, p in self.model.named_parameters()
        )
        self._caesar_final_params = OrderedDict(
            (n, p.data.cpu().clone()) for n, p in self.model.named_parameters()
        )

    def _compress_gradient(self, grad: OrderedDict, theta_u: float) -> OrderedDict:
        """Top-K sparsification: retain top (1-theta_u) fraction by magnitude, zero the rest."""
        compressed = OrderedDict()
        for name, g in grad.items():
            flat = g.view(-1)
            n = flat.numel()
            n_keep = max(1, int(n * (1.0 - theta_u)))
            _, order = torch.sort(flat.abs(), descending=True)
            mask = torch.zeros(n)
            mask[order[:n_keep]] = 1.0
            compressed[name] = (flat * mask).view(g.shape)
        return compressed

    def package(self) -> Dict[str, Any]:
        result = super().package()
        result["_caesar_gradient"] = self._compress_gradient(
            self._caesar_gradient, self._caesar_theta_u
        )
        # Carried for server bookkeeping (stateless sim); not part of paper wire protocol
        result["_caesar_final_params"] = self._caesar_final_params
        return result
