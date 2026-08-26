# -*- coding: utf-8 -*-
"""Shared sparse-training protocol and mask operations."""

import math
import re
from argparse import Namespace
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .tFL import tFL, tFL_Client

_DEFAULT_IGNORE = [r".*\.bias$", r".*bn.*", r".*ln.*"]
_DEFAULT_IGNORE_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)


class spFLShared:
    """Stateless mask math shared by sparse servers and workers."""

    @staticmethod
    def f_decay(t: int, alpha: float, T_end: int) -> float:
        """Return the cosine topology-adjustment fraction."""
        if T_end <= 0:
            raise ValueError("T_end must be positive")
        # Paper schedule: alpha_t = alpha / 2 * (1 + cos(t * pi / T_end)).
        return alpha / 2 * (1 + math.cos(min(max(t, 0), T_end) * math.pi / T_end))

    @staticmethod
    def adjustment_round(
        current_iter: int,
        delta_T: int,
        T_end: int,
        include_zero: bool = False,
    ) -> bool:
        """Return whether this round changes the sparse topology."""
        if delta_T <= 0:
            raise ValueError("delta_T must be positive")
        return (
            (include_zero or current_iter > 0)
            and current_iter % delta_T == 0
            and current_iter <= T_end
        )

    @staticmethod
    def clone_mask(mask_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Clone masks so serial workers cannot mutate server state."""
        return {
            name: mask.detach().cpu().bool().clone() for name, mask in mask_dict.items()
        }

    @staticmethod
    def get_sparse_layers(
        model: nn.Module,
        ignore_patterns: Optional[List[str]] = None,
    ) -> Set[str]:
        """Return parameter names eligible for unstructured sparsity."""
        patterns = _DEFAULT_IGNORE if ignore_patterns is None else ignore_patterns
        sparse = {
            name
            for name, _ in model.named_parameters()
            if all(re.match(pattern, name) is None for pattern in patterns)
        }
        for module_name, module in model.named_modules():
            if isinstance(module, _DEFAULT_IGNORE_TYPES):
                prefix = f"{module_name}." if module_name else ""
                sparse = {name for name in sparse if not name.startswith(prefix)}
        return sparse

    @staticmethod
    def _erk_densities(
        shape_dict: Mapping[str, Tuple[int, ...]],
        sparse_set: Set[str],
        target_count: int,
        is_kernel: bool,
    ) -> Dict[str, float]:
        """Allocate an ER/ERK density while preserving the target count."""
        dense_layers: Set[str] = set()
        while True:
            remaining = sparse_set - dense_layers
            if not remaining:
                return {name: 1.0 for name in sparse_set}
            raw = {
                name: (
                    sum(shape_dict[name]) / math.prod(shape_dict[name])
                    if is_kernel
                    else sum(shape_dict[name][:2]) / math.prod(shape_dict[name][:2])
                )
                for name in remaining
            }
            dense_count = sum(math.prod(shape_dict[name]) for name in dense_layers)
            divisor = sum(raw[name] * math.prod(shape_dict[name]) for name in remaining)
            epsilon = (target_count - dense_count) / divisor
            overfull = {name for name in remaining if epsilon * raw[name] > 1}
            if not overfull:
                return {
                    name: 1.0 if name in dense_layers else epsilon * raw[name]
                    for name in sparse_set
                }
            dense_layers.update(overfull)

    @staticmethod
    def generate_layer_density_dict(
        model: nn.Module,
        target_density: float,
        strategy: str = "ERK_magnitude",
    ) -> Dict[str, float]:
        """Return per-layer densities for a global density budget."""
        if not 0 < target_density <= 1:
            raise ValueError("target_density must be in (0, 1]")
        distribution, _ = strategy.split("_", maxsplit=1)
        shape_dict = {
            name: tuple(param.shape) for name, param in model.named_parameters()
        }
        sparse_set = spFLShared.get_sparse_layers(model=model)
        if not sparse_set:
            return {}
        total = sum(math.prod(shape) for shape in shape_dict.values())
        dense_count = sum(
            math.prod(shape_dict[name]) for name in shape_dict if name not in sparse_set
        )
        target_count = int(target_density * total) - dense_count
        sparse_count = total - dense_count
        if not 0 < target_count <= sparse_count:
            raise ValueError("target_density is incompatible with the dense parameters")
        if distribution == "uniform":
            density = target_count / sparse_count
            return {name: density for name in sparse_set}
        if distribution in ("ER", "ERK"):
            return spFLShared._erk_densities(
                shape_dict=shape_dict,
                sparse_set=sparse_set,
                target_count=target_count,
                is_kernel=distribution == "ERK",
            )
        raise ValueError(f"Unknown density strategy: {distribution}")

    @staticmethod
    def init_mask(
        model: nn.Module,
        target_density: float,
        strategy: str = "ERK_magnitude",
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Initialize a boolean mask at the requested density."""
        _, pruning = strategy.split("_", maxsplit=1)
        densities = spFLShared.generate_layer_density_dict(
            model=model,
            target_density=target_density,
            strategy=strategy,
        )
        masks: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if name not in densities:
                continue
            keep = max(1, min(param.numel(), int(param.numel() * densities[name])))
            if pruning in ("magnitude", "mag"):
                indices = torch.topk(
                    input=param.detach().abs().flatten(),
                    k=keep,
                    largest=True,
                    sorted=False,
                ).indices.cpu()
            elif pruning == "random":
                indices = torch.randperm(param.numel())[:keep]
            else:
                raise ValueError(f"Unknown pruning strategy: {pruning}")
            mask = torch.zeros(param.numel(), dtype=torch.bool)
            mask[indices] = True
            masks[name] = mask.view(param.shape)
        return masks, densities

    @staticmethod
    @torch.no_grad()
    def apply_mask(
        model: nn.Module,
        mask_dict: Mapping[str, torch.Tensor],
    ) -> None:
        """Zero parameters outside the supplied topology."""
        for name, param in model.named_parameters():
            if name in mask_dict:
                param.mul_(mask_dict[name].to(device=param.device))

    @staticmethod
    def swap_mask(
        parameters: Mapping[str, torch.Tensor],
        gradients: Mapping[str, torch.Tensor],
        mask_dict: Mapping[str, torch.Tensor],
        fraction: float,
        names: Optional[Set[str]] = None,
        prune_indices: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Prune active weights and grow from the original inactive set."""
        updated = spFLShared.clone_mask(mask_dict=mask_dict)
        selected = set(mask_dict) if names is None else names
        for name in selected:
            if name not in mask_dict or name not in parameters or name not in gradients:
                continue
            original = mask_dict[name].flatten().cpu().bool()
            active = original.nonzero(as_tuple=False).flatten()
            inactive = (~original).nonzero(as_tuple=False).flatten()
            count = min(int(fraction * active.numel()), inactive.numel())
            if prune_indices is not None and name in prune_indices:
                prune = prune_indices[name].flatten().long()[:count]
                count = min(prune.numel(), inactive.numel())
                prune = prune[:count]
            elif count > 0:
                weights = parameters[name].detach().cpu().abs().flatten()
                prune = active[
                    torch.topk(
                        input=weights[active],
                        k=count,
                        largest=False,
                        sorted=False,
                    ).indices
                ]
            else:
                continue
            gradient = gradients[name].detach().cpu().abs().flatten()
            grow = inactive[
                torch.topk(
                    input=gradient[inactive],
                    k=count,
                    largest=True,
                    sorted=False,
                ).indices
            ]
            flat = original.clone()
            flat[prune] = False
            flat[grow] = True
            updated[name] = flat.view_as(mask_dict[name])
        return updated

    @staticmethod
    def union_masks(
        masks: Sequence[Mapping[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Return the coordinate-wise union of client masks."""
        if not masks:
            raise ValueError("at least one mask is required")
        return {
            name: torch.stack([mask[name].detach().cpu().bool() for mask in masks]).any(
                dim=0
            )
            for name in masks[0]
        }

    @staticmethod
    def magnitude_reprune(
        parameters: Mapping[str, torch.Tensor],
        candidate_mask: Mapping[str, torch.Tensor],
        layer_densities: Mapping[str, float],
    ) -> Dict[str, torch.Tensor]:
        """Magnitude-prune a mask union back to its layer budgets."""
        result: Dict[str, torch.Tensor] = {}
        for name, density in layer_densities.items():
            candidates = (
                candidate_mask[name]
                .flatten()
                .cpu()
                .bool()
                .nonzero(as_tuple=False)
                .flatten()
            )
            keep = max(1, int(parameters[name].numel() * density))
            if candidates.numel() < keep:
                raise ValueError(
                    f"mask union for {name} has fewer than {keep} candidates"
                )
            weights = parameters[name].detach().cpu().abs().flatten()
            chosen = candidates[
                torch.topk(
                    input=weights[candidates],
                    k=keep,
                    largest=True,
                    sorted=False,
                ).indices
            ]
            mask = torch.zeros(parameters[name].numel(), dtype=torch.bool)
            mask[chosen] = True
            result[name] = mask.view_as(candidate_mask[name])
        return result

    @staticmethod
    def sparse_weighted_mean(
        models: Sequence[Mapping[str, torch.Tensor]],
        masks: Sequence[Mapping[str, torch.Tensor]],
        weights: Sequence[float],
        fallback_model: Optional[Mapping[str, torch.Tensor]] = None,
        fallback_mask: Optional[Mapping[str, torch.Tensor]] = None,
        fallback_weight: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Average each sparse coordinate over clients that retained it."""
        if not models or len(models) != len(masks) or len(models) != len(weights):
            raise ValueError(
                "models, masks, and weights must have equal non-zero length"
            )
        model_weights = torch.as_tensor(weights, dtype=torch.float64)
        if not torch.isfinite(model_weights).all() or (model_weights < 0).any():
            raise ValueError("weights must be finite and non-negative")
        total_weight = float(model_weights.sum()) + fallback_weight
        if total_weight <= 0:
            raise ValueError("weights must have a positive sum")
        averaged = type(models[0])()
        sparse_names = set().union(*(mask.keys() for mask in masks))
        for name in models[0]:
            values = torch.stack([model[name] for model in models])
            if not values.is_floating_point() and not values.is_complex():
                averaged[name] = values[0].clone()
                continue
            dtype = (
                torch.float32
                if values.dtype in (torch.float16, torch.bfloat16)
                else values.dtype
            )
            values = values.to(dtype=dtype)
            weight_view = model_weights.to(dtype=dtype).view(
                (-1,) + (1,) * (values.ndim - 1)
            )
            if name not in sparse_names:
                numerator = (values * weight_view).sum(dim=0)
                if fallback_model is not None and fallback_weight > 0:
                    numerator.add_(
                        fallback_model[name].to(dtype=dtype), alpha=fallback_weight
                    )
                averaged[name] = (numerator / total_weight).to(models[0][name].dtype)
                continue
            coordinate_masks = torch.stack(
                [mask[name].cpu().bool() for mask in masks]
            ).to(dtype=dtype)
            numerator = (values * coordinate_masks * weight_view).sum(dim=0)
            denominator = (coordinate_masks * weight_view).sum(dim=0)
            if (
                fallback_model is not None
                and fallback_mask is not None
                and fallback_weight > 0
            ):
                old_mask = fallback_mask[name].cpu().bool().to(dtype=dtype)
                numerator.add_(
                    fallback_model[name].to(dtype=dtype) * old_mask,
                    alpha=fallback_weight,
                )
                denominator.add_(old_mask, alpha=fallback_weight)
            averaged[name] = torch.where(
                denominator > 0,
                numerator / denominator.clamp_min(torch.finfo(dtype).tiny),
                torch.zeros_like(numerator),
            ).to(models[0][name].dtype)
        return averaged


class spFL(spFLShared, tFL):
    """Server base for a single globally managed sparse topology."""

    optional = {
        "target_density": 0.5,
        "delta_T": 50,
        "T_end": 500,
        "adjust_alpha": 0.3,
        "pruning_strategy": "ERK_magnitude",
    }

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        self._sp_mask_dict: Dict[str, torch.Tensor] = {}
        self._sp_layer_density: Dict[str, float] = {}

    def _sp_is_adj(self) -> bool:
        return self.adjustment_round(
            current_iter=self.current_iter,
            delta_T=self.delta_T,
            T_end=self.T_end,
        )

    def _sp_init_mask(self) -> None:
        self._sp_mask_dict, self._sp_layer_density = self.init_mask(
            model=self.model,
            target_density=self.target_density,
            strategy=self.pruning_strategy,
        )
        self._sp_commit_mask()

    def _sp_parameter_state(self) -> "OrderedDict[str, torch.Tensor]":
        return OrderedDict(
            (name, param.detach().cpu().clone())
            for name, param in self.model.named_parameters()
        )

    def _sp_commit_mask(self) -> None:
        self.apply_mask(model=self.model, mask_dict=self._sp_mask_dict)
        self._commit_global(new_params=self._sp_parameter_state())

    def package(self, client_id: int) -> Dict[str, Any]:
        if not self._sp_mask_dict:
            self._sp_init_mask()
        package = super().package(client_id=client_id)
        package["_sp_mask_dict"] = self.clone_mask(mask_dict=self._sp_mask_dict)
        package["__wire__"] += ("_sp_mask_dict",)
        return package

    def aggregate_client_updates(self, packages: Mapping[int, Dict[str, Any]]) -> None:
        super().aggregate_client_updates(packages=packages)
        if self._sp_is_adj():
            self._sp_update_mask(packages=packages)
        self._sp_commit_mask()

    def _sp_update_mask(self, packages: Mapping[int, Dict[str, Any]]) -> None:
        """Keep the topology fixed unless a strategy overrides this hook."""


class spFL_Client(spFLShared, tFL_Client):
    """Reusable worker that trains only coordinates in the supplied mask."""

    def set_parameters(self, package: Dict[str, Any]) -> None:
        super().set_parameters(package=package)
        self._sp_mask_dict = self.clone_mask(mask_dict=package["_sp_mask_dict"])
        self._sp_is_adj = self.adjustment_round(
            current_iter=self.current_iter,
            delta_T=self.delta_T,
            T_end=self.T_end,
        )

    def _train_masked_epochs(
        self,
        dataloader: DataLoader,
        epochs: int,
        offload_after_epoch: bool,
    ) -> None:
        """Train while preventing optimizer state from reviving masked weights."""
        for _ in range(epochs):
            self.model.to(self.device)
            self._move_optimizer_state_to_param_devices(optimizer=self.optimizer)
            self.model.train()
            for batch_x, batch_y, x_mark, y_mark in dataloader:
                self.optimizer.zero_grad(set_to_none=True)
                batch_x = batch_x.to(
                    device=self.device, dtype=torch.float32, non_blocking=True
                )
                batch_y = batch_y.to(
                    device=self.device, dtype=torch.float32, non_blocking=True
                )
                x_mark = x_mark.to(device=self.device, non_blocking=True)
                y_mark = y_mark.to(device=self.device, non_blocking=True)
                prediction = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
                self.loss(prediction, batch_y).backward()
                for name, param in self.model.named_parameters():
                    if name in self._sp_mask_dict and param.grad is not None:
                        param.grad.mul_(
                            self._sp_mask_dict[name].to(device=param.grad.device)
                        )
                self.optimizer.step()
                self.step_scheduler_batch(
                    scheduler=self.scheduler,
                    batch_data=batch_x,
                )
                self.apply_mask(model=self.model, mask_dict=self._sp_mask_dict)
            self.step_scheduler_epoch(scheduler=self.scheduler)
            if offload_after_epoch:
                self.model.to("cpu")

    def fit(self) -> None:
        self._set_worker_seed(seed=self._loader_seed(dataset_type="train"))
        loader = self.load_train_data()
        self.initialize_scheduler(steps_per_epoch=len(loader))
        self.apply_mask(model=self.model, mask_dict=self._sp_mask_dict)
        self._train_masked_epochs(
            dataloader=loader,
            epochs=self.epochs,
            offload_after_epoch=self.efficiency == "low",
        )
        if self.efficiency == "med":
            self.model.to("cpu")

    def _collect_gradients(
        self,
        names: Optional[Set[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return one minibatch's dense gradient for selected parameters."""
        selected = set(self._sp_mask_dict) if names is None else names
        self.model.to(self.device)
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        for batch_x, batch_y, x_mark, y_mark in self.load_train_data():
            batch_x = batch_x.to(device=self.device, dtype=torch.float32)
            batch_y = batch_y.to(device=self.device, dtype=torch.float32)
            x_mark = x_mark.to(device=self.device)
            y_mark = y_mark.to(device=self.device)
            prediction = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
            self.loss(prediction, batch_y).backward()
            break
        # ponytail: PyTorch still materializes dense selected-layer gradients;
        # replace this only when the TSF models expose sparse backward kernels.
        gradients = {
            name: (
                param.grad.detach().cpu().clone()
                if param.grad is not None
                else torch.zeros_like(param, device="cpu")
            )
            for name, param in self.model.named_parameters()
            if name in selected
        }
        self.model.zero_grad(set_to_none=True)
        return gradients

    def _package_sparse_extra(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        package = super().package()
        if extra:
            package["_sp_extra"] = extra
            package["__wire__"] += ("_sp_extra",)
        return package
