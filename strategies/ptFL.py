"""Physical partial-training base for model-heterogeneous FL.

The server keeps each client's parameter-index manifest.  Clients receive and
return only dense narrow parameter tensors; they never echo global indices.
Subclasses define the index schedule, while model-family adapters define the
meaning of a hidden unit and all tensor axes coupled to it.
"""

from __future__ import annotations

import copy
import math
from argparse import Namespace
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, TypeAlias

import torch
from torch import nn

from .base import SharedMethods
from .tFL import tFL, tFL_Client

IndexTuple: TypeAlias = tuple[torch.Tensor, ...]
Manifest: TypeAlias = OrderedDict[str, IndexTuple]
IndexSelector: TypeAlias = Callable[[str, int, int], torch.Tensor]


@dataclass(frozen=True)
class PTPlan:
    """Server-only extraction plan for one client and one round."""

    manifest: Manifest
    retained_widths: tuple[int, ...]
    is_degenerate: bool = False


class ptFL_SharedMethods:
    """Static physical-submodel rules shared by ptFL servers and clients."""

    _PT_OUTPUT_ONLY_MODELS = frozenset(
        {"DLinear", "DishLinear", "Linear", "NLinear", "RLinear"}
    )
    _PT_RECURRENT_MODELS = frozenset({"GRU", "LSTM"})
    _PT_INPUT_WEIGHT_PREFIXES = ("W_i", "weight_ih")
    _PT_RECURRENT_WEIGHT_PREFIXES = ("W_h", "weight_hh")
    _PT_BIAS_PREFIXES = ("b_", "bias_ih", "bias_hh")

    @staticmethod
    def _pt_validate_capacity(capacity: float) -> None:
        if not 0.0 < capacity <= 1.0:
            raise ValueError(f"capacity must be in (0, 1], got {capacity}")

    @staticmethod
    def _pt_retained_width(full_width: int, capacity: float) -> int:
        """Return the manuscript's exact ``floor(beta * K)`` width."""

        ptFL_SharedMethods._pt_validate_capacity(capacity)
        width = math.floor(capacity * full_width)
        if width < 1:
            raise ValueError(
                "capacity selects zero units under floor(beta * K): "
                f"capacity={capacity}, K={full_width}"
            )
        return width

    @staticmethod
    def _pt_full_indices(parameter: torch.Tensor) -> IndexTuple:
        return tuple(torch.arange(size, dtype=torch.long) for size in parameter.shape)

    @staticmethod
    def _pt_validate_selected_indices(
        indices: torch.Tensor,
        full_width: int,
        retained: int,
        group_name: str,
    ) -> torch.Tensor:
        indices = indices.detach().cpu().to(dtype=torch.long).flatten()
        if indices.numel() != retained:
            raise ValueError(
                f"{group_name} selected {indices.numel()} units; expected {retained}"
            )
        if indices.unique().numel() != retained:
            raise ValueError(f"{group_name} contains duplicate unit indices")
        if bool(((indices < 0) | (indices >= full_width)).any()):
            raise IndexError(f"{group_name} has indices outside [0, {full_width})")
        return indices

    @staticmethod
    def _pt_extract_parameters(
        parameters: Mapping[str, torch.Tensor],
        manifest: Mapping[str, IndexTuple],
    ) -> OrderedDict[str, torch.Tensor]:
        """Extract dense tensors described by a server-side manifest."""

        if set(parameters) != set(manifest):
            missing = sorted(set(parameters) - set(manifest))
            extra = sorted(set(manifest) - set(parameters))
            raise KeyError(f"manifest mismatch: missing={missing}, extra={extra}")

        result: OrderedDict[str, torch.Tensor] = OrderedDict()
        for name, parameter in parameters.items():
            selected = parameter
            indices = manifest[name]
            if len(indices) != parameter.dim():
                raise ValueError(
                    f"{name}: {len(indices)} index axes for {parameter.dim()}D tensor"
                )
            for dimension, index in enumerate(indices):
                selected = selected.index_select(dimension, index)
            result[name] = selected.detach().cpu().clone()
        return result

    @staticmethod
    def _pt_reject_buffers(model: nn.Module) -> None:
        buffers = [name for name, _ in model.named_buffers()]
        if buffers:
            raise NotImplementedError(
                "ptFL must declare buffer semantics explicitly; "
                f"unsupported buffers: {buffers}"
            )

    @staticmethod
    def _pt_adapter_kind(model_name: str, model: nn.Module | None = None) -> str:
        """Resolve only model families with explicitly declared axis semantics."""

        if model_name in ptFL_SharedMethods._PT_RECURRENT_MODELS:
            return "recurrent_stack"
        if model_name in ptFL_SharedMethods._PT_OUTPUT_ONLY_MODELS:
            return "output_only"
        if model is not None and hasattr(model, "cells") and hasattr(model, "fc_pred"):
            return "recurrent_stack"
        raise NotImplementedError(
            f"no physical partial-training adapter for model {model_name!r}; "
            "add model-family rules to ptFL rather than slicing tensors by shape"
        )

    @staticmethod
    def _pt_expanded_rows(
        parameter: torch.Tensor,
        hidden_size: int,
        selected: torch.Tensor,
        parameter_name: str,
    ) -> torch.Tensor:
        if parameter.shape[0] % hidden_size != 0:
            raise ValueError(
                f"{parameter_name}: first dimension {parameter.shape[0]} is not "
                f"a whole number of hidden blocks of size {hidden_size}"
            )
        blocks = parameter.shape[0] // hidden_size
        return torch.cat([selected + block * hidden_size for block in range(blocks)])

    @staticmethod
    def _pt_build_output_only_plan(model: nn.Module, capacity: float) -> PTPlan:
        ptFL_SharedMethods._pt_validate_capacity(capacity)
        ptFL_SharedMethods._pt_reject_buffers(model)
        manifest: Manifest = OrderedDict(
            (name, ptFL_SharedMethods._pt_full_indices(parameter))
            for name, parameter in model.named_parameters()
        )
        return PTPlan(manifest=manifest, retained_widths=(), is_degenerate=True)

    @staticmethod
    def _pt_build_recurrent_plan(
        model: nn.Module,
        capacity: float,
        selector: IndexSelector,
    ) -> PTPlan:
        """Map independent per-layer cell selections to every coupled axis."""

        ptFL_SharedMethods._pt_reject_buffers(model)
        if not hasattr(model, "cells") or not isinstance(model.cells, nn.ModuleList):
            raise TypeError("recurrent ptFL requires model.cells as ModuleList")
        if not hasattr(model, "fc_pred") or not isinstance(model.fc_pred, nn.Linear):
            raise TypeError("recurrent ptFL requires a fixed nn.Linear fc_pred head")
        if not model.cells:
            raise ValueError("recurrent stack must contain at least one cell")

        hidden_sizes = tuple(int(cell.hidden_size) for cell in model.cells)
        retained_widths = tuple(
            ptFL_SharedMethods._pt_retained_width(hidden_size, capacity)
            for hidden_size in hidden_sizes
        )
        if len(set(retained_widths)) != 1:
            raise NotImplementedError(
                "current recurrent constructors expose one hidden_size for all layers"
            )

        selected_by_layer = []
        for layer, (hidden_size, width) in enumerate(
            zip(hidden_sizes, retained_widths)
        ):
            selected_by_layer.append(
                ptFL_SharedMethods._pt_validate_selected_indices(
                    selector(f"cells.{layer}", hidden_size, width),
                    hidden_size,
                    width,
                    f"cells.{layer}",
                )
            )

        manifest: Manifest = OrderedDict()
        for layer, cell in enumerate(model.cells):
            hidden_size = hidden_sizes[layer]
            selected = selected_by_layer[layer]
            previous = selected_by_layer[layer - 1] if layer > 0 else None
            for local_name, parameter in cell.named_parameters(recurse=False):
                full_name = f"cells.{layer}.{local_name}"
                rows = ptFL_SharedMethods._pt_expanded_rows(
                    parameter,
                    hidden_size,
                    selected,
                    full_name,
                )
                if local_name.startswith(ptFL_SharedMethods._PT_INPUT_WEIGHT_PREFIXES):
                    if parameter.dim() != 2:
                        raise ValueError(f"{full_name} must be a matrix")
                    columns = (
                        torch.arange(parameter.shape[1], dtype=torch.long)
                        if previous is None
                        else previous
                    )
                    manifest[full_name] = (rows, columns)
                elif local_name.startswith(
                    ptFL_SharedMethods._PT_RECURRENT_WEIGHT_PREFIXES
                ):
                    if parameter.dim() != 2:
                        raise ValueError(f"{full_name} must be a matrix")
                    manifest[full_name] = (rows, selected)
                elif local_name.startswith(ptFL_SharedMethods._PT_BIAS_PREFIXES):
                    if parameter.dim() != 1:
                        raise ValueError(f"{full_name} must be a vector")
                    manifest[full_name] = (rows,)
                else:
                    raise NotImplementedError(
                        f"no recurrent cell-axis rule for {full_name}"
                    )

        last_selected = selected_by_layer[-1]
        for local_name, parameter in model.fc_pred.named_parameters(recurse=False):
            full_name = f"fc_pred.{local_name}"
            if local_name == "weight":
                manifest[full_name] = (
                    torch.arange(parameter.shape[0], dtype=torch.long),
                    last_selected,
                )
            elif local_name == "bias":
                manifest[full_name] = (
                    torch.arange(parameter.shape[0], dtype=torch.long),
                )
            else:
                raise NotImplementedError(
                    f"unsupported forecast-head tensor {full_name}"
                )

        parameter_names = {name for name, _ in model.named_parameters()}
        if set(manifest) != parameter_names:
            missing = sorted(parameter_names - set(manifest))
            extra = sorted(set(manifest) - parameter_names)
            raise NotImplementedError(
                f"recurrent ptFL coverage mismatch: missing={missing}, extra={extra}"
            )
        return PTPlan(
            manifest=manifest,
            retained_widths=retained_widths,
            is_degenerate=False,
        )

    @staticmethod
    def _pt_build_plan(
        model_name: str,
        model: nn.Module,
        capacity: float,
        selector: IndexSelector,
    ) -> PTPlan:
        adapter_kind = ptFL_SharedMethods._pt_adapter_kind(model_name, model)
        if adapter_kind == "output_only":
            return ptFL_SharedMethods._pt_build_output_only_plan(model, capacity)
        return ptFL_SharedMethods._pt_build_recurrent_plan(model, capacity, selector)

    @staticmethod
    def _pt_build_model(configs: Namespace) -> nn.Module:
        model_class = SharedMethods._get_objective_function("models", configs.model)
        return model_class(configs=configs)

    @staticmethod
    def _pt_build_client_model(configs: Namespace, capacity: float) -> nn.Module:
        adapter_kind = ptFL_SharedMethods._pt_adapter_kind(configs.model)
        narrow_configs = copy.deepcopy(configs)
        if adapter_kind == "recurrent_stack":
            narrow_configs.hidden_size = ptFL_SharedMethods._pt_retained_width(
                int(configs.hidden_size), capacity
            )
        else:
            ptFL_SharedMethods._pt_validate_capacity(capacity)
        return ptFL_SharedMethods._pt_build_model(narrow_configs)


class ptFL(ptFL_SharedMethods, tFL):
    """Base server for dense physical partial-training strategies."""

    optional = {"capacity": "1.0"}

    @classmethod
    def args_update(cls, parser) -> None:
        parser.add_argument(
            "--capacity",
            type=str,
            default=None,
            help="Comma-separated retained hidden-width ratios in (0, 1]",
        )

    def __init__(self, configs, times) -> None:
        super().__init__(configs, times)
        self._pt_capacities = self._parse_capacities(self.capacity)
        self._pt_adapter_kind(self.configs.model, self.model)
        self._pt_pending_manifests: Dict[int, Manifest] = {}
        self._pt_warned_degenerate = False

    @staticmethod
    def _parse_capacities(raw: str) -> tuple[float, ...]:
        try:
            capacities = tuple(float(value.strip()) for value in raw.split(","))
        except (AttributeError, ValueError) as error:
            raise ValueError(f"invalid capacity list {raw!r}") from error
        if not capacities:
            raise ValueError("capacity list cannot be empty")
        invalid = [value for value in capacities if not 0.0 < value <= 1.0]
        if invalid:
            raise ValueError(f"capacities must be in (0, 1], got {invalid}")
        return capacities

    def _pt_capacity_for_client(self, client_id: int) -> float:
        return self._pt_capacities[client_id % len(self._pt_capacities)]

    def _pt_select_indices(
        self,
        group_name: str,
        full_width: int,
        retained: int,
        client_id: int,
    ) -> torch.Tensor:
        raise NotImplementedError("ptFL subclasses must define an index schedule")

    def package(self, client_id: int) -> dict:
        capacity = self._pt_capacity_for_client(client_id)
        plan = self._pt_build_plan(
            self.configs.model,
            self.model,
            capacity,
            lambda group, width, retained: self._pt_select_indices(
                group,
                width,
                retained,
                client_id,
            ),
        )
        if plan.is_degenerate and not self._pt_warned_degenerate:
            self.logger.warning(
                "%s has no hidden-width axis; partial training is full-width",
                self.configs.model,
            )
            self._pt_warned_degenerate = True

        self._pt_pending_manifests[client_id] = plan.manifest
        submodel = self._pt_extract_parameters(self.public_model_params, plan.manifest)
        return {
            "__wire__": ("regular_model_params", "capacity"),
            "client_id": client_id,
            "current_iter": self.current_iter,
            "regular_model_params": submodel,
            "capacity": capacity,
        }

    @staticmethod
    def _pt_index_grid(indices: IndexTuple) -> IndexTuple:
        if not indices:
            return ()
        return tuple(torch.meshgrid(*indices, indexing="ij"))

    def aggregate_client_updates(self, packages: "OrderedDict[int, dict]") -> None:
        accum = OrderedDict(
            (name, torch.zeros_like(parameter))
            for name, parameter in self.public_model_params.items()
        )
        counts = OrderedDict(
            (
                name,
                torch.zeros_like(parameter, dtype=torch.float32),
            )
            for name, parameter in self.public_model_params.items()
        )

        for client_id, package in packages.items():
            if client_id not in self._pt_pending_manifests:
                raise KeyError(
                    f"missing server-side PT manifest for client {client_id}"
                )
            manifest = self._pt_pending_manifests.pop(client_id)
            local_parameters = package["regular_model_params"]
            if set(local_parameters) != set(self.public_model_params):
                missing = sorted(set(self.public_model_params) - set(local_parameters))
                extra = sorted(set(local_parameters) - set(self.public_model_params))
                raise KeyError(
                    f"client {client_id} parameter mismatch: "
                    f"missing={missing}, extra={extra}"
                )

            for name, server_parameter in self.public_model_params.items():
                indices = manifest[name]
                local = local_parameters[name].to(dtype=server_parameter.dtype)
                expected_shape = tuple(index.numel() for index in indices)
                if tuple(local.shape) != expected_shape:
                    raise ValueError(
                        f"client {client_id} {name}: shape {tuple(local.shape)}; "
                        f"expected {expected_shape}"
                    )
                grid = self._pt_index_grid(indices)
                if not grid:
                    accum[name].add_(local)
                    counts[name].add_(1.0)
                    continue
                accum[name].index_put_(grid, local, accumulate=True)
                counts[name].index_put_(
                    grid,
                    torch.ones_like(local, dtype=torch.float32),
                    accumulate=True,
                )

        new_parameters = OrderedDict()
        for name, original in self.public_model_params.items():
            count = counts[name]
            updated = accum[name] / count.clamp_min(1.0).to(accum[name].dtype)
            new_parameters[name] = torch.where(count > 0, updated, original)
        self._commit_global(new_parameters)


class ptFL_Client(ptFL_SharedMethods, tFL_Client):
    """Stateless client that rebuilds its physical narrow model each round."""

    def set_parameters(self, package: Mapping[str, object]) -> None:
        self.id = int(package["client_id"])
        self.current_iter = int(package["current_iter"])
        self._load_private(self.id)

        capacity = float(package["capacity"])
        self.model = self._pt_build_client_model(self.configs, capacity)
        self.model.load_state_dict(package["regular_model_params"], strict=True)
        self.optimizer = self._build("optimizers", self.configs.optimizer)(
            params=self.model.parameters(),
            configs=self.configs,
        )
        self.scheduler = self._build("schedulers", self.configs.scheduler)(
            optimizer=self.optimizer,
            configs=self.configs,
        )
        self.regular_params_name = [name for name, _ in self.model.named_parameters()]
        self.personal_params_name = []

    def package(self) -> dict:
        regular = OrderedDict(
            (name, parameter.detach().cpu().clone())
            for name, parameter in self.model.named_parameters()
        )
        return {
            "__wire__": ("regular_model_params",),
            "client_id": self.id,
            "regular_model_params": regular,
            "personal_model_params": {},
            "optimizer_state": {},
            "scheduler_state": {},
            "score": self.train_samples,
        }

    def evaluate_global(
        self,
        client_id: int,
        global_params: "OrderedDict[str, torch.Tensor]",
        dataset_type: str,
        current_iter: int,
    ) -> float:
        self.id = client_id
        self._load_private(client_id)
        model_class = self._build("models", self.configs.model)
        self.model = model_class(configs=self.configs)
        return super().evaluate_global(
            client_id,
            global_params,
            dataset_type,
            current_iter,
        )
