# -*- coding: utf-8 -*-
"""Caesar - A Low-deviation Compression Approach for Efficient Federated Learning."""

from argparse import Namespace
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

import torch

from .tFL import tFL, tFL_Client

_BYTES_PER_FLOAT32 = 4
_BYTES_PER_INT64 = 8
_BYTES_PER_MB = 1024**2


class CaesarShared:
    """Caesar wire-size math shared by server and worker."""

    @staticmethod
    def compressed_downlink_bytes(
        compressed: Optional[Dict[str, Any]],
    ) -> float:
        """Return the compressed-model wire size in bytes."""
        if compressed is None:
            return 0.0
        return float(
            sum(
                len(data["full_idx"]) * (_BYTES_PER_FLOAT32 + _BYTES_PER_INT64)
                + len(data["comp_idx"])
                + 2 * _BYTES_PER_FLOAT32
                for data in compressed.values()
            )
        )

    @staticmethod
    def sparse_gradient_bytes(
        gradients: Mapping[str, torch.Tensor],
    ) -> float:
        """Return the COO gradient wire size in bytes."""
        return float(
            sum(
                gradient.count_nonzero().item()
                * (_BYTES_PER_INT64 + _BYTES_PER_FLOAT32)
                for gradient in gradients.values()
            )
        )


class Caesar(CaesarShared, tFL):
    """Caesar server - staleness-aware download + importance-ranked gradient upload."""

    optional = {
        "theta_d_max": 0.6,
        "theta_u_min": 0.1,
        "theta_u_max": 0.6,
    }

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        self._caesar_last_round: Dict[int, int] = {}
        self._caesar_importance: Dict[int, float] = {}
        self._caesar_upload_ratio: Dict[int, float] = {}
        self._caesar_prev_params: Dict[int, Optional[OrderedDict]] = {}
        self._caesar_ratio_iter: int = -1
        self._caesar_downlink_mb: Dict[int, float] = (
            {}
        )  # true wire size, bypasses _dispatch overwrite

    def _caesar_init_importance(self, scores: Mapping[int, float]) -> None:
        self._caesar_importance.update(scores)
        max_s = max(self._caesar_importance.values()) or 1.0
        for cid in self._caesar_importance:
            self._caesar_importance[cid] /= max_s

    def _caesar_update_upload_ratios(self, selected: List[int]) -> None:
        n = self.num_clients
        ranked = sorted(
            selected, key=lambda c: self._caesar_importance.get(c, 0.0), reverse=True
        )
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
            self._caesar_update_upload_ratios(selected=self.selected_clients)
            self._caesar_ratio_iter = self.current_iter

        t = max(self.current_iter, 1)
        delta = t - self._caesar_last_round.get(client_id, 0)
        theta_d = (1.0 - delta / t) * self.theta_d_max
        theta_u = self._caesar_upload_ratio.get(client_id, self.theta_u_min)

        pkg = super().package(client_id=client_id)
        compressed = self._caesar_compress(theta_d=theta_d)
        pkg["_caesar_compressed"] = compressed
        pkg["_caesar_prev_params"] = self._caesar_prev_params.get(client_id)
        pkg["_caesar_theta_u"] = theta_u

        # Record true downlink wire size (compressed dict when theta_d>0, full model otherwise)
        if compressed is not None:
            self._caesar_downlink_mb[client_id] = (
                self.compressed_downlink_bytes(compressed=compressed) / _BYTES_PER_MB
            )
        else:
            self._caesar_downlink_mb[client_id] = self.get_size(
                obj=self.public_model_params
            )
        return pkg

    def _compute_send_mb(
        self,
        packages: Mapping[int, Dict[str, Any]],
    ) -> tuple[Dict[int, float], float]:
        # Uplink: COO sparse gradient size per client (excludes _caesar_final_params)
        uplink = {}
        for cid, pkg in packages.items():
            grad = pkg.get("_caesar_gradient", {})
            uplink[cid] = self.sparse_gradient_bytes(gradients=grad) / _BYTES_PER_MB

        # Downlink: sum of per-client true wire sizes (set in package() before _dispatch overwrites)
        downlink = sum(
            self._caesar_downlink_mb.get(cid, 0.0) for cid in self.selected_clients
        )
        return uplink, downlink

    def aggregate_client_updates(self, packages: "OrderedDict[int, dict]") -> None:
        bootstrap = not self._caesar_importance
        gradients = []
        scores = {}
        selected = []
        for client_id, package in packages.items():
            gradients.append(package["_caesar_gradient"])
            self._caesar_prev_params[client_id] = package["_caesar_final_params"]
            self._caesar_last_round[client_id] = self.current_iter
            if bootstrap:
                scores[client_id] = float(package["score"])
                selected.append(client_id)

        # Bootstrap importance on first round.
        if bootstrap:
            self._caesar_init_importance(scores=scores)
            self._caesar_update_upload_ratios(selected=selected)
            self._caesar_ratio_iter = self.current_iter

        avg_grad = self.mean_models(models=gradients)

        # w^{t+1} = w^t - avg_gradient
        new_params = OrderedDict()
        for name in self.public_model_params:
            delta = avg_grad.get(name, torch.zeros_like(self.public_model_params[name]))
            new_params[name] = (self.public_model_params[name].float() - delta).to(
                self.public_model_params[name].dtype
            )
        self._commit_global(new_params=new_params)


class Caesar_Client(CaesarShared, tFL_Client):
    """Caesar client - model recovery + gradient compression."""

    def set_parameters(self, package: Dict[str, Any]) -> None:
        super().set_parameters(package=package)
        self._caesar_theta_u = package["_caesar_theta_u"]
        compressed = package["_caesar_compressed"]
        if compressed is not None:
            self._caesar_recover(
                compressed=compressed,
                prev_params=package["_caesar_prev_params"],
            )

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
                    recovered = torch.where(
                        sign_ok & abs_ok, prev_comp, signs * cdata["avg_abs"]
                    )
                else:
                    recovered = signs * cdata["avg_abs"]
                flat[comp_idx] = recovered
            state[name] = flat.view(cdata["shape"])
        self.model.load_state_dict(state, strict=False)

    def fit(self) -> None:
        self._set_worker_seed(seed=self._loader_seed(dataset_type="train"))

        # Snapshot starting params to compute gradient after training
        init_params = OrderedDict(
            (n, p.data.cpu().clone()) for n, p in self.model.named_parameters()
        )

        loader = self.load_train_data()
        self.initialize_scheduler(steps_per_epoch=len(loader))
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
            grad=self._caesar_gradient,
            theta_u=self._caesar_theta_u,
        )
        # Carried for server bookkeeping (stateless sim); not part of paper wire protocol
        result["_caesar_final_params"] = self._caesar_final_params
        return result
