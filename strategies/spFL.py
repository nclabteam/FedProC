# -*- coding: utf-8 -*-
"""spFL - Sparse/Pruning Federated Learning base class.

Server manages a global binary mask dict; clients apply it during local training.
Adjustment rounds (every ``delta_T`` rounds up to ``T_end``) trigger mask updates
via the ``_sp_update_mask`` hook that subclasses override.

Pruning utilities (f_decay, init_mask, apply_mask, sparse_update_step, ...) live
as static methods on spFL so any subclass can call them via ``self.method()``.
"""
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

from .tFL import tFL, tFL_Client

# ---------------------------------------------------------------------------
# Private constants
# ---------------------------------------------------------------------------

_DEFAULT_IGNORE = [r".*\.bias$", r".*bn.*", r".*ln.*"]
_DEFAULT_IGNORE_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)


class spFL(tFL):
    """Sparse FL server base.

    Manages the global binary mask and exposes pruning utilities as static methods.
    Subclasses override ``_sp_update_mask`` to implement their specific mask policy.
    """

    optional = {
        "target_density": 0.5,
        "delta_T": 50,
        "T_end": 500,
        "adjust_alpha": 0.3,
        "pruning_strategy": "ERK_magnitude",
    }

    def __init__(self, configs, times):
        super().__init__(configs, times)
        self._sp_mask_dict: Dict[str, torch.Tensor] = {}
        self._sp_layer_density: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Round control
    # ------------------------------------------------------------------

    def _sp_is_adj(self) -> bool:
        return (
            self.current_iter > 0
            and self.current_iter % self.delta_T == 0
            and self.current_iter <= self.T_end
        )

    def _sp_init_mask(self) -> None:
        self._sp_mask_dict, self._sp_layer_density = self.init_mask(
            self.model, self.target_density, self.pruning_strategy
        )
        self.apply_mask(self.model, self._sp_mask_dict)

    # ------------------------------------------------------------------
    # FL protocol
    # ------------------------------------------------------------------

    def package(self, client_id: int) -> Dict[str, Any]:
        if not self._sp_mask_dict:
            self._sp_init_mask()
        pkg = super().package(client_id)
        pkg["_sp_mask_dict"] = self._sp_mask_dict
        pkg["_sp_is_adj"] = self._sp_is_adj()
        pkg["_sp_t"] = self.current_iter
        pkg["_sp_T_end"] = self.T_end
        pkg["_sp_alpha"] = self.adjust_alpha
        return pkg

    def aggregate_client_updates(self, packages: Dict[int, Any]) -> None:
        super().aggregate_client_updates(packages)
        self.apply_mask(self.model, self._sp_mask_dict)
        if self._sp_is_adj():
            self._sp_update_mask(packages)

    def _sp_update_mask(self, packages: Dict[int, Any]) -> None:
        """Override in subclasses to implement strategy-specific mask update."""

    # ------------------------------------------------------------------
    # Static pruning utilities
    # ------------------------------------------------------------------

    @staticmethod
    def f_decay(t: int, alpha: float, T_end: int) -> float:
        """Cosine decay: alpha/2 * (1 + cos(t*pi/T_end))."""
        return alpha / 2 * (1 + np.cos(t * np.pi / T_end))

    @staticmethod
    def get_sparse_layers(
        model: nn.Module,
        ignore_patterns: Optional[List[str]] = None,
    ) -> Set[str]:
        if ignore_patterns is None:
            ignore_patterns = _DEFAULT_IGNORE
        sparse = {n for n, _ in model.named_parameters()}
        for pat in ignore_patterns:
            sparse = {n for n in sparse if re.match(pat, n) is None}
        for name, module in model.named_modules():
            if isinstance(module, _DEFAULT_IGNORE_TYPES):
                sparse = {n for n in sparse if not n.startswith(name)}
        return sparse

    @staticmethod
    def _erk_densities(
        shape_dict: Dict[str, tuple],
        sparse_set: Set[str],
        target_density: float,
    ) -> Dict[str, float]:
        total = sum(np.prod(s) for s in shape_dict.values())
        dense_total = sum(np.prod(shape_dict[n]) for n in shape_dict if n not in sparse_set)
        assert int(target_density * total) > dense_total, "target_density too small"

        eps, dense_layers = 1.0, set()
        valid = False
        while not valid:
            divisor, rhs, raw = 0.0, 0.0, {}
            for name in sparse_set:
                if name in dense_layers:
                    rhs -= int(np.prod(shape_dict[name]) * (1 - target_density))
                    continue
                n = np.prod(shape_dict[name])
                rhs += int(n * target_density)
                p = np.sum(shape_dict[name]) / n
                raw[name] = p
                divisor += p * n
            eps = rhs / divisor
            max_p = max(raw.values())
            if eps * max_p > 1:
                for nm, pv in raw.items():
                    if pv == max_p:
                        dense_layers.add(nm)
            else:
                valid = True
        return {
            n: 1.0 if n in dense_layers else min(eps * raw[n], 1.0)
            for n in sparse_set
        }

    @staticmethod
    def generate_layer_density_dict(
        model: nn.Module,
        target_density: float,
        strategy: str = "ERK_magnitude",
    ) -> Dict[str, float]:
        dist, _ = strategy.split("_")
        shape_dict = {n: tuple(p.shape) for n, p in model.named_parameters()}
        sparse_set = spFL.get_sparse_layers(model)
        total = sum(np.prod(s) for s in shape_dict.values())
        dense_total = sum(np.prod(shape_dict[n]) for n in shape_dict if n not in sparse_set)

        if dist == "uniform":
            sparse_total = total - dense_total
            d = (int(target_density * total) - dense_total) / sparse_total
            return {n: float(d) for n in sparse_set}
        if dist in ("ER", "ERK"):
            return spFL._erk_densities(shape_dict, sparse_set, target_density)
        raise ValueError(f"Unknown density strategy: {dist}")

    @staticmethod
    def init_mask(
        model: nn.Module,
        target_density: float,
        strategy: str = "ERK_magnitude",
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        _, prune_strat = strategy.split("_")
        layer_density = spFL.generate_layer_density_dict(model, target_density, strategy)
        mask_dict: Dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if name not in layer_density:
                continue
            density = layer_density[name]
            n = param.numel()
            num_keep = max(1, int(n * density))

            if prune_strat in ("magnitude", "mag"):
                _, idx = torch.sort(param.data.abs().view(-1), descending=True)
                mask = torch.zeros(n, dtype=param.dtype)
                mask[idx[:num_keep]] = 1.0
                mask_dict[name] = mask.view(param.shape)
            elif prune_strat == "random":
                idx = torch.randperm(n)
                mask = torch.zeros(n, dtype=param.dtype)
                mask[idx[:num_keep]] = 1.0
                mask_dict[name] = mask.view(param.shape)
            else:
                raise ValueError(f"Unknown pruning strategy: {prune_strat}")

        return mask_dict, layer_density

    @staticmethod
    @torch.no_grad()
    def apply_mask(
        model: nn.Module,
        mask_dict: Dict[str, torch.Tensor],
    ) -> None:
        for name, param in model.named_parameters():
            if name in mask_dict:
                param.data.mul_(mask_dict[name].to(param.device))

    @staticmethod
    def sparse_update_step(
        model: nn.Module,
        gradients: Dict[str, torch.Tensor],
        mask_dict: Dict[str, torch.Tensor],
        t: int,
        T_end: int,
        alpha: float,
    ) -> Dict[str, torch.Tensor]:
        """Prune k smallest active by magnitude; grow k largest inactive by gradient."""
        for name, param in model.named_parameters():
            if name not in mask_dict:
                continue
            mask = mask_dict[name]
            active_num = int((mask == 1).sum().item())
            k = int(spFL.f_decay(t, alpha, T_end) * active_num)
            if k <= 0:
                continue

            flat_w = param.data.abs().view(-1)
            flat_m = mask.view(-1)

            active_idx = (flat_m == 1).nonzero(as_tuple=False).view(-1)
            _, prune_rel = torch.topk(flat_w[active_idx], k, largest=False)
            flat_m[active_idx[prune_rel]] = 0.0

            inactive_idx = (flat_m == 0).nonzero(as_tuple=False).view(-1)
            flat_g = gradients[name].abs().view(-1).to(flat_w.device)
            _, grow_rel = torch.topk(flat_g[inactive_idx], min(k, len(inactive_idx)), largest=True)
            flat_m[inactive_idx[grow_rel]] = 1.0

            mask_dict[name] = flat_m.view(mask.shape)

        return mask_dict

    @staticmethod
    def sparse_pruning_step(
        model: nn.Module,
        mask_dict: Dict[str, torch.Tensor],
        t: int,
        T_end: int,
        alpha: float,
    ) -> Dict[str, torch.Tensor]:
        for name, param in model.named_parameters():
            if name not in mask_dict:
                continue
            mask = mask_dict[name]
            active_num = int((mask == 1).sum().item())
            k = int(spFL.f_decay(t, alpha, T_end) * active_num)
            if k <= 0:
                continue
            flat_w = param.data.abs().view(-1)
            flat_m = mask.view(-1)
            active_idx = (flat_m == 1).nonzero(as_tuple=False).view(-1)
            _, prune_rel = torch.topk(flat_w[active_idx], k, largest=False)
            flat_m[active_idx[prune_rel]] = 0.0
            mask_dict[name] = flat_m.view(mask.shape)
        return mask_dict

    @staticmethod
    def sparse_growing_step(
        model: nn.Module,
        gradients: Dict[str, torch.Tensor],
        mask_dict: Dict[str, torch.Tensor],
        layer_density_dict: Dict[str, float],
    ) -> Dict[str, torch.Tensor]:
        for name, param in model.named_parameters():
            if name not in layer_density_dict or name not in mask_dict:
                continue
            mask = mask_dict[name]
            target_k = max(1, int(param.numel() * layer_density_dict[name]))
            current_k = int((mask == 1).sum().item())
            k = max(0, target_k - current_k)
            if k <= 0:
                continue
            flat_m = mask.view(-1)
            flat_g = gradients[name].abs().view(-1).to(flat_m.device)
            inactive_idx = (flat_m == 0).nonzero(as_tuple=False).view(-1)
            _, grow_rel = torch.topk(flat_g[inactive_idx], min(k, len(inactive_idx)), largest=True)
            flat_m[inactive_idx[grow_rel]] = 1.0
            mask_dict[name] = flat_m.view(mask.shape)
        return mask_dict

    @staticmethod
    def union_masks(masks_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Logical OR across a list of mask dicts."""
        result: Dict[str, torch.Tensor] = {}
        for masks in masks_list:
            for name, m in masks.items():
                if name not in result:
                    result[name] = m.clone().bool()
                else:
                    result[name] = result[name] | m.bool()
        return {n: m.float() for n, m in result.items()}

    @staticmethod
    def magnitude_reprune(
        model: nn.Module,
        union_mask: Dict[str, torch.Tensor],
        layer_density_dict: Dict[str, float],
    ) -> Dict[str, torch.Tensor]:
        """Magnitude-prune the unioned mask back to target density per layer."""
        new_mask = {}
        for name, param in model.named_parameters():
            if name not in layer_density_dict:
                continue
            density = layer_density_dict[name]
            num_keep = max(1, int(param.numel() * density))
            masked_w = param.data.abs() * union_mask[name].to(param.device)
            _, idx = torch.sort(masked_w.view(-1), descending=True)
            mask = torch.zeros_like(param.data.view(-1))
            mask[idx[:num_keep]] = 1.0
            new_mask[name] = mask.view(param.shape)
        return new_mask


class spFL_Client(tFL_Client):
    """Sparse FL client base. Applies mask during local training."""

    # Static utilities forwarded from spFL so subclasses can call self.method()
    f_decay = staticmethod(spFL.f_decay)
    apply_mask = staticmethod(spFL.apply_mask)
    sparse_update_step = staticmethod(spFL.sparse_update_step)
    sparse_pruning_step = staticmethod(spFL.sparse_pruning_step)
    sparse_growing_step = staticmethod(spFL.sparse_growing_step)
    union_masks = staticmethod(spFL.union_masks)
    magnitude_reprune = staticmethod(spFL.magnitude_reprune)

    def set_parameters(self, package: Dict[str, Any]) -> None:
        super().set_parameters(package)
        self._sp_mask_dict = package["_sp_mask_dict"]
        self._sp_is_adj = package["_sp_is_adj"]
        self._sp_t = package["_sp_t"]
        self._sp_T_end = package["_sp_T_end"]
        self._sp_alpha = package["_sp_alpha"]

    def fit(self) -> None:
        from .base import SharedMethods
        SharedMethods._set_worker_seed(self._loader_seed("train"))
        loader = self.load_train_data()
        offload = self.efficiency == "low"
        self.apply_mask(self.model, self._sp_mask_dict)
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
            self.apply_mask(self.model, self._sp_mask_dict)
        if self.efficiency == "med":
            self.model.to("cpu")

    def _collect_gradients(self) -> Dict[str, torch.Tensor]:
        """Single-batch backward pass for gradient growth signal."""
        self.model.to(self.device)
        self.model.train()
        self.model.zero_grad()
        for batch_x, batch_y, x_mark, y_mark in self.load_train_data():
            batch_x = batch_x.to(self.device, dtype=torch.float32)
            batch_y = batch_y.to(self.device, dtype=torch.float32)
            x_mark = x_mark.to(self.device)
            y_mark = y_mark.to(self.device)
            pred = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
            loss = self.loss(pred, batch_y)
            loss.backward()
            break
        grads = {
            n: p.grad.detach().cpu().clone() if p.grad is not None else torch.zeros(p.shape)
            for n, p in self.model.named_parameters()
        }
        self.model.zero_grad()
        return grads
