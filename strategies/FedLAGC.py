"""Layer-Adaptive submodel extraction with Gradient Correction."""

from __future__ import annotations

import math
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, TypeAlias

import torch
from torch import nn

from .ptFL import (
    ParameterState,
    TrainableSpec,
    ptFLLocalMetric,
    ptFLUpdate,
    ptFLUpdate_Client,
)
from .spFL import spFLShared

SparseParameter: TypeAlias = tuple[torch.Tensor | None, torch.Tensor]
SparseState: TypeAlias = OrderedDict[str, SparseParameter]


class FedLAGCShared(spFLShared):
    """FedLAGC math and transport shared by server and client."""

    _FEDLAGC_NORMS = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.LayerNorm,
        nn.GroupNorm,
    )

    @staticmethod
    def _fedlagc_is_bias(parameter_name: str) -> bool:
        leaf = parameter_name.rsplit(".", maxsplit=1)[-1]
        return leaf == "bias" or leaf.startswith(("b_", "bias_"))

    @classmethod
    def _fedlagc_layout(
        cls, model: nn.Module
    ) -> tuple[set[str], OrderedDict[str, tuple[str, ...]]]:
        groups = cls._pt_parameter_groups(model=model)
        if not groups:
            raise ValueError("FedLAGC requires at least one parameterized layer")
        modules = dict(model.named_modules())
        edge_layers = {next(iter(groups)), next(reversed(groups))}
        critical: set[str] = set()
        prunable: OrderedDict[str, tuple[str, ...]] = OrderedDict()
        for layer_name, parameter_names in groups.items():
            if layer_name in edge_layers or isinstance(
                modules[layer_name], cls._FEDLAGC_NORMS
            ):
                critical.update(parameter_names)
                continue
            layer_prunable = tuple(
                name
                for name in parameter_names
                if not cls._fedlagc_is_bias(parameter_name=name)
            )
            critical.update(set(parameter_names) - set(layer_prunable))
            if layer_prunable:
                prunable[layer_name] = layer_prunable
        return critical, prunable

    @staticmethod
    def _fedlagc_allocate_counts(
        mean_importance: torch.Tensor,
        parameter_counts: torch.Tensor,
        budget: int,
    ) -> torch.Tensor:
        if (
            mean_importance.ndim != 1
            or parameter_counts.shape != mean_importance.shape
            or budget < 0
        ):
            raise ValueError("FedLAGC allocation inputs are inconsistent")
        if bool((mean_importance < 0).any()) or bool((parameter_counts <= 0).any()):
            raise ValueError("importance must be non-negative and counts positive")
        if budget > int(parameter_counts.sum().item()):
            raise ValueError("FedLAGC budget exceeds the prunable parameters")
        if budget == 0:
            return torch.zeros_like(parameter_counts, dtype=torch.long)

        # Paper: S~_l = log(1 + S_l) / sum_j log(1 + S_j).
        normalized = torch.log1p(mean_importance.to(dtype=torch.float64))
        normalizer = normalized.sum()
        if not bool(normalizer > 0):
            raise ValueError("FedLAGC cannot rank an all-zero model")
        normalized.div_(normalizer)

        # Paper: r_l = S~_l (d_n - d~) / sum_j(S~_j d_j).
        counts = parameter_counts.to(dtype=torch.float64)
        quotas = normalized * budget * counts / torch.dot(normalized, counts)
        if bool((quotas > counts + 1e-9).any()):
            raise ValueError(
                "FedLAGC paper allocation exceeds a layer; use a smaller capacity"
            )

        allocated = quotas.floor().to(dtype=torch.long)
        residual = budget - int(allocated.sum().item())
        if residual:
            order = torch.argsort(
                quotas - allocated,
                descending=True,
                stable=True,
            )
            allocated[order[:residual]] += 1
        return allocated

    @classmethod
    def _fedlagc_mask(
        cls,
        model: nn.Module,
        capacity: float,
    ) -> ParameterState:
        cls._pt_validate_capacity(capacity=capacity)
        parameters = OrderedDict(
            (name, parameter.detach().cpu())
            for name, parameter in model.named_parameters()
        )
        masks: ParameterState = OrderedDict(
            (name, torch.zeros_like(parameter, dtype=torch.bool))
            for name, parameter in parameters.items()
        )
        if capacity == 1.0:
            for mask in masks.values():
                mask.fill_(True)
            return masks

        critical, prunable = cls._fedlagc_layout(model=model)
        for name in critical:
            masks[name].fill_(True)
        total = sum(parameter.numel() for parameter in parameters.values())
        target = math.floor(capacity * total)
        critical_count = sum(parameters[name].numel() for name in critical)
        if target <= critical_count:
            raise ValueError(
                "FedLAGC capacity must exceed the critical-component fraction "
                f"{critical_count / total:.6f}"
            )
        if not prunable:
            raise ValueError("FedLAGC found no prunable layers")

        # Only the weights are prunable, so only they are ranked and counted.
        layer_values = [
            torch.cat([parameters[name].abs().flatten() for name in names])
            for names in prunable.values()
        ]
        # The importance mean, though, is over "the total number of parameters
        # in the l-th layer": a prunable layer's bias is fully retained but is
        # still one of that layer's parameters, so it counts toward S_l.
        groups = cls._pt_parameter_groups(model=model)
        layer_importance = torch.stack(
            [
                torch.cat(
                    [parameters[name].abs().flatten() for name in groups[layer_name]]
                )
                .to(dtype=torch.float64)
                .mean()
                for layer_name in prunable
            ]
        )
        retained = cls._fedlagc_allocate_counts(
            mean_importance=layer_importance,
            parameter_counts=torch.tensor(
                [values.numel() for values in layer_values], dtype=torch.long
            ),
            budget=target - critical_count,
        )

        for names, values, keep in zip(
            prunable.values(), layer_values, retained.tolist()
        ):
            selected = torch.zeros(values.numel(), dtype=torch.bool)
            if keep:
                indices = torch.topk(
                    input=values,
                    k=keep,
                    largest=True,
                    sorted=False,
                ).indices
                selected[indices] = True
            offset = 0
            for name in names:
                size = parameters[name].numel()
                masks[name] = selected[offset : offset + size].view_as(parameters[name])
                offset += size
        return masks

    @staticmethod
    def _fedlagc_compress(
        parameters: Mapping[str, torch.Tensor],
        masks: Mapping[str, torch.Tensor],
    ) -> SparseState:
        if set(parameters) != set(masks):
            raise KeyError("FedLAGC parameters and masks must have identical names")
        sparse: SparseState = OrderedDict()
        for name, parameter in parameters.items():
            value = parameter.detach().cpu()
            mask = masks[name].detach().cpu().bool().flatten()
            if mask.numel() != value.numel():
                raise ValueError(f"FedLAGC mask shape mismatch for {name}")
            if bool(mask.all()):
                sparse[name] = (None, value.clone())
            elif bool(mask.any()):
                indices = mask.nonzero(as_tuple=False).flatten()
                index_dtype = torch.int32 if value.numel() <= 2**31 else torch.int64
                sparse[name] = (
                    indices.to(dtype=index_dtype),
                    value.flatten()[indices].clone(),
                )
        return sparse

    @staticmethod
    def _fedlagc_expand(
        model: nn.Module,
        sparse: Mapping[str, SparseParameter],
    ) -> tuple[ParameterState, ParameterState]:
        parameters = OrderedDict(model.named_parameters())
        unknown = set(sparse) - set(parameters)
        if unknown:
            raise KeyError(f"FedLAGC payload has unknown parameters: {sorted(unknown)}")
        state: ParameterState = OrderedDict()
        masks: ParameterState = OrderedDict()
        for name, parameter in parameters.items():
            value = torch.zeros_like(parameter, device="cpu")
            mask = torch.zeros_like(parameter, dtype=torch.bool, device="cpu")
            if name in sparse:
                indices, selected = sparse[name]
                selected = selected.detach().cpu().to(dtype=value.dtype)
                if indices is None:
                    if selected.shape != value.shape:
                        raise ValueError(f"FedLAGC dense shape mismatch for {name}")
                    value.copy_(selected)
                    mask.fill_(True)
                else:
                    indices = indices.detach().cpu().long().flatten()
                    if (
                        selected.numel() != indices.numel()
                        or indices.unique().numel() != indices.numel()
                        or bool(((indices < 0) | (indices >= value.numel())).any())
                    ):
                        raise ValueError(f"FedLAGC sparse indices are invalid for {name}")
                    value.flatten()[indices] = selected.flatten()
                    mask.flatten()[indices] = True
            state[name] = value
            masks[name] = mask
        return state, masks

    @classmethod
    def _fedlagc_thresholds(
        cls,
        model: nn.Module,
        masks: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        _critical, prunable = cls._fedlagc_layout(model=model)
        parameters = dict(model.named_parameters())
        thresholds: dict[str, torch.Tensor] = {}
        for names in prunable.values():
            selected = [
                parameters[name].detach().abs()[
                    masks[name].to(device=parameters[name].device)
                ]
                for name in names
                if bool(masks[name].any())
            ]
            if not selected:
                continue
            threshold = torch.cat([value.flatten() for value in selected]).min()
            thresholds.update((name, threshold) for name in names)
        return thresholds

    @staticmethod
    def _fedlagc_correction_active(current_iter: int, iterations: int) -> bool:
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        # Paper heuristic: h(t) = 1 when t < T / 4, otherwise 0.  ``current_iter``
        # is a 0-indexed integer, so the exact realization of the real-valued
        # comparison ``t < T / 4`` is ``t < ceil(T / 4)``.
        return current_iter < math.ceil(iterations / 4)

    @staticmethod
    def _fedlagc_correction(
        parameters: Mapping[str, torch.Tensor],
        correction: Mapping[str, torch.Tensor] | None,
    ) -> ParameterState:
        if correction is not None and set(correction) != set(parameters):
            raise KeyError("FedLAGC correction must cover every model parameter")
        result: ParameterState = OrderedDict()
        for name, parameter in parameters.items():
            value = (
                torch.zeros_like(parameter, device="cpu")
                if correction is None
                else correction[name].detach().cpu().clone()
            )
            if value.shape != parameter.shape:
                raise ValueError(f"FedLAGC correction shape mismatch for {name}")
            result[name] = value
        return result


class FedLAGC(ptFLLocalMetric, FedLAGCShared, ptFLUpdate):
    """FedLAGC server."""

    optional = {
        "capacity": "1.0,0.25,0.0625,0.015625",
        "fedlagc_beta": 0.1,
    }

    @classmethod
    def args_update(cls, parser: ArgumentParser) -> None:
        parser.add_argument("--capacity", type=str, default=None)

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        self._fedlagc_capacities = self._pt_parse_ratios(
            raw=self.capacity,
            option_name="capacity",
        )
        if self.fedlagc_beta <= 0:
            raise ValueError("fedlagc_beta must be positive")
        # The paper's submodel is ``theta * M``: coordinates outside a client's
        # mask are zero in its model, not held at the global value.
        self._pt_init_local_metric()

    def _pt_update_spec(self, client_id: int) -> TrainableSpec:
        del client_id
        raise NotImplementedError("FedLAGC uses parameter masks, not row specs")

    def _fedlagc_client_mask(self, client_id: int) -> "OrderedDict[str, torch.Tensor]":
        return self._fedlagc_mask(
            model=self.model,
            capacity=self._fedlagc_capacities[
                client_id % len(self._fedlagc_capacities)
            ],
        )

    def _pt_resource_level_params(self, client_id: int) -> ParameterState:
        # The paper's submodel is "theta * M": coordinates outside the mask are
        # zero in the client's model, not held at the global value.
        mask = self._fedlagc_client_mask(client_id=client_id)
        return OrderedDict(
            (name, parameter * mask[name].to(parameter.dtype))
            for name, parameter in self.public_model_params.items()
        )

    def package(self, client_id: int) -> dict[str, Any]:
        mask = self._fedlagc_client_mask(client_id=client_id)
        self._pt_pending_update_masks[client_id] = self.clone_mask(mask_dict=mask)
        if "personalization" in self._pt_local_metrics:
            # "theta * M" is this client's model until it receives another one.
            self._pt_last_submodel[client_id] = OrderedDict(
                (name, parameter * mask[name].to(parameter.dtype))
                for name, parameter in self.public_model_params.items()
            )
        stored = self.clients_personal_model_params[client_id].get(
            "fedlagc_correction"
        )
        correction = (
            self._fedlagc_correction(
                parameters=self.public_model_params,
                correction=None,
            )
            if stored is None
            else stored
        )
        return {
            "__wire__": ("fedlagc_submodel",),
            "client_id": client_id,
            "current_iter": self.current_iter,
            "fedlagc_submodel": self._fedlagc_compress(
                parameters=self.public_model_params,
                masks=mask,
            ),
            "fedlagc_correction": correction,
            "personal_model_params": {},
            "optimizer_state": self.client_optimizer_states[client_id],
            "scheduler_state": self.client_scheduler_states[client_id],
        }


class FedLAGC_Client(FedLAGCShared, ptFLUpdate_Client):
    """FedLAGC client."""

    def __init__(self, configs: Namespace, times: int, device: str) -> None:
        super().__init__(configs=configs, times=times, device=device)
        if self.fedlagc_beta <= 0:
            raise ValueError("fedlagc_beta must be positive")

    def _pt_resolve_update_mask(
        self, package: Mapping[str, Any]
    ) -> ParameterState:
        del package
        return self._fedlagc_payload_mask

    def set_parameters(self, package: dict[str, Any]) -> None:
        state, self._fedlagc_payload_mask = self._fedlagc_expand(
            model=self.model,
            sparse=package["fedlagc_submodel"],
        )
        materialized = dict(package)
        materialized["regular_model_params"] = state
        materialized["personal_model_params"] = {}
        super().set_parameters(package=materialized)
        self._fedlagc_parameters = dict(self.model.named_parameters())
        self._fedlagc_threshold = self._fedlagc_thresholds(
            model=self.model,
            masks=self._pt_trainable_mask,
        )
        self._fedlagc_lambda = self._fedlagc_correction(
            parameters=OrderedDict(self.model.named_parameters()),
            correction=package["fedlagc_correction"],
        )
        self._fedlagc_use_correction = self._fedlagc_correction_active(
            current_iter=self.current_iter,
            iterations=self.iterations,
        )

    def _pt_mask_gradient(self, name: str, gradient: torch.Tensor) -> torch.Tensor:
        masked = super()._pt_mask_gradient(name=name, gradient=gradient)
        threshold = self._fedlagc_threshold.get(name)
        if threshold is not None:
            magnitude = self._fedlagc_parameters[name].detach().abs()
            threshold = threshold.to(device=magnitude.device, dtype=magnitude.dtype)
            denominator = (magnitude + threshold).square()
            # Paper STE: 1 + 2 |theta| theta~ / (|theta| + theta~)^2.
            multiplier = torch.where(
                denominator > 0,
                1 + 2 * magnitude * threshold / denominator.clamp_min(
                    torch.finfo(denominator.dtype).tiny
                ),
                torch.ones_like(denominator),
            )
            masked.mul_(multiplier)
        if self._fedlagc_use_correction:
            mask = self._pt_trainable_mask[name]
            correction = self._fedlagc_lambda[name].to(
                device=masked.device,
                dtype=masked.dtype,
            )
            # Paper: g~ = (g - h(t) lambda) element-wise multiplied by M.
            masked.sub_(correction * mask.to(device=masked.device))
        return masked

    def package(self) -> dict[str, Any]:
        package = super().package()
        for name, difference in package["model_params_diff"].items():
            mask = self._pt_trainable_mask[name].detach().cpu().bool()
            # Paper: lambda_{t+1} = lambda_t + beta (theta_{t+1}-theta_{t,0}) M.
            self._fedlagc_lambda[name][mask] -= (
                self.fedlagc_beta * difference.to(self._fedlagc_lambda[name])
            )
        package["personal_model_params"] = {
            "fedlagc_correction": self._fedlagc_lambda
        }
        return package
