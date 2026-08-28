"""Physical partial-training base for model-heterogeneous FL."""

from __future__ import annotations

import copy
import math
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Mapping, TypeAlias

import numpy as np
import torch
from torch import nn

from .base import SharedMethods
from .tFL import tFL, tFL_Client

IndexTuple: TypeAlias = tuple[torch.Tensor, ...]
Manifest: TypeAlias = OrderedDict[str, IndexTuple]
ParameterState: TypeAlias = OrderedDict[str, torch.Tensor]
IndexSelector: TypeAlias = Callable[[str, int, int], torch.Tensor]
TrainableSpec: TypeAlias = OrderedDict[str, tuple[int, ...] | None]


@dataclass(frozen=True)
class PTPlan:
    """Server-only extraction plan for one client and one round."""

    manifest: Manifest
    retained_widths: tuple[int, ...]
    is_degenerate: bool = False


class ptFLShared(SharedMethods):
    """Physical-submodel rules shared by ptFL servers and clients."""

    _PT_OUTPUT_ONLY_MODELS = frozenset(
        {"DLinear", "DishLinear", "Linear", "NLinear", "RLinear"}
    )
    _PT_RECURRENT_MODELS = frozenset({"GRU", "LSTM"})
    _PT_INPUT_WEIGHT_PREFIXES = ("W_i", "weight_ih")
    _PT_RECURRENT_WEIGHT_PREFIXES = ("W_h", "weight_hh")
    _PT_BIAS_PREFIXES = ("b_", "bias_ih", "bias_hh")

    @staticmethod
    def _pt_parse_ratios(raw: str, option_name: str) -> tuple[float, ...]:
        try:
            ratios = tuple(float(value.strip()) for value in raw.split(","))
        except (AttributeError, ValueError) as error:
            raise ValueError(f"invalid {option_name} list {raw!r}") from error
        if not ratios or any(not 0.0 < value <= 1.0 for value in ratios):
            raise ValueError(f"{option_name} must contain values in (0, 1]")
        return ratios

    @staticmethod
    def _pt_validate_capacity(capacity: float) -> None:
        if not 0.0 < capacity <= 1.0:
            raise ValueError(f"capacity must be in (0, 1], got {capacity}")

    @classmethod
    def _pt_retained_width(cls, full_width: int, capacity: float) -> int:
        """Return the manuscript's exact ``floor(beta * K)`` width."""

        cls._pt_validate_capacity(capacity=capacity)
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
    def _pt_validate_depth_layers(
        depth_layers: tuple[int, ...], num_layers: int
    ) -> tuple[int, ...]:
        """Validate retained layers; layer 0 owns the non-hidden input axis."""

        layers = tuple(sorted(set(int(layer) for layer in depth_layers)))
        if not layers:
            raise ValueError("depth_layers cannot be empty")
        if layers[0] != 0 or layers[-1] >= num_layers:
            raise ValueError(
                f"depth_layers must be a subset of [0, {num_layers}) containing "
                f"layer 0, got {layers}"
            )
        return layers

    @staticmethod
    def _pt_cell(model: nn.Module, layer: int) -> nn.Module:
        cells = model.cells
        return cells[str(layer)] if isinstance(cells, nn.ModuleDict) else cells[layer]

    @staticmethod
    def _pt_rename_cell_layer(name: str, layer_map: Mapping[int, int]) -> str:
        """Rename ``cells.<old>.<leaf>`` via ``layer_map``; passthrough otherwise."""

        if not name.startswith("cells."):
            return name
        _, old_index, leaf = name.split(".", 2)
        return f"cells.{layer_map[int(old_index)]}.{leaf}"

    @classmethod
    def _pt_rename_state(
        cls,
        state: Mapping[str, torch.Tensor],
        layer_map: Mapping[int, int],
    ) -> ParameterState:
        return OrderedDict(
            (
                cls._pt_rename_cell_layer(name=name, layer_map=layer_map),
                value,
            )
            for name, value in state.items()
        )

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
    ) -> ParameterState:
        """Extract manifest tensors while preserving omitted depth layers."""

        unknown = set(manifest) - set(parameters)
        if unknown:
            raise KeyError(f"manifest references unknown parameters: {sorted(unknown)}")

        result: ParameterState = OrderedDict()
        for name, indices in manifest.items():
            parameter = parameters[name]
            selected = parameter
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

    @classmethod
    def _pt_adapter_kind(cls, model_name: str, model: nn.Module | None = None) -> str:
        """Resolve only model families with explicitly declared axis semantics."""

        if model_name in cls._PT_RECURRENT_MODELS:
            return "recurrent_stack"
        if model_name in cls._PT_OUTPUT_ONLY_MODELS:
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

    @classmethod
    def _pt_build_output_only_plan(cls, model: nn.Module, capacity: float) -> PTPlan:
        cls._pt_validate_capacity(capacity=capacity)
        if capacity < 1.0:
            # These models expose no hidden-width axis, so a fractional capacity
            # would silently train at full width and report a partial-training
            # result that no PT-FL paper describes. Reject the pairing instead.
            raise ValueError(
                f"{type(model).__name__} has no hidden-width axis; physical "
                f"partial training requires capacity=1.0, got {capacity}"
            )
        cls._pt_reject_buffers(model=model)
        manifest: Manifest = OrderedDict(
            (name, cls._pt_full_indices(parameter=parameter))
            for name, parameter in model.named_parameters()
        )
        return PTPlan(manifest=manifest, retained_widths=(), is_degenerate=True)

    @classmethod
    def _pt_build_recurrent_plan(
        cls,
        model: nn.Module,
        capacity: float,
        selector: IndexSelector,
        depth_layers: tuple[int, ...] | None = None,
    ) -> PTPlan:
        """Map retained server layers and units to every coupled tensor axis."""

        cls._pt_reject_buffers(model=model)
        if not hasattr(model, "cells") or not isinstance(
            model.cells, (nn.ModuleList, nn.ModuleDict)
        ):
            raise TypeError(
                "recurrent ptFL requires model.cells as ModuleList or ModuleDict"
            )
        if not hasattr(model, "fc_pred") or not isinstance(model.fc_pred, nn.Linear):
            raise TypeError("recurrent ptFL requires a fixed nn.Linear fc_pred head")
        if not model.cells:
            raise ValueError("recurrent stack must contain at least one cell")

        num_layers = len(model.cells)
        if depth_layers is None:
            depth_layers = tuple(range(num_layers))
        depth_layers = cls._pt_validate_depth_layers(
            depth_layers=depth_layers, num_layers=num_layers
        )

        hidden_sizes = tuple(
            int(cls._pt_cell(model=model, layer=server_layer).hidden_size)
            for server_layer in depth_layers
        )
        retained_widths = tuple(
            cls._pt_retained_width(full_width=hidden_size, capacity=capacity)
            for hidden_size in hidden_sizes
        )
        if len(set(retained_widths)) != 1:
            raise NotImplementedError(
                "current recurrent constructors expose one hidden_size for all layers"
            )

        selected_by_layer = []
        for client_layer, server_layer in enumerate(depth_layers):
            hidden_size = hidden_sizes[client_layer]
            width = retained_widths[client_layer]
            selected_by_layer.append(
                cls._pt_validate_selected_indices(
                    indices=selector(
                        group_name=f"cells.{server_layer}",
                        full_width=hidden_size,
                        retained=width,
                    ),
                    full_width=hidden_size,
                    retained=width,
                    group_name=f"cells.{server_layer}",
                )
            )

        manifest: Manifest = OrderedDict()
        for client_layer, server_layer in enumerate(depth_layers):
            cell = cls._pt_cell(model=model, layer=server_layer)
            hidden_size = hidden_sizes[client_layer]
            selected = selected_by_layer[client_layer]
            previous = selected_by_layer[client_layer - 1] if client_layer > 0 else None
            for local_name, parameter in cell.named_parameters(recurse=False):
                full_name = f"cells.{server_layer}.{local_name}"
                rows = cls._pt_expanded_rows(
                    parameter=parameter,
                    hidden_size=hidden_size,
                    selected=selected,
                    parameter_name=full_name,
                )
                if local_name.startswith(cls._PT_INPUT_WEIGHT_PREFIXES):
                    if parameter.dim() != 2:
                        raise ValueError(f"{full_name} must be a matrix")
                    columns = (
                        torch.arange(parameter.shape[1], dtype=torch.long)
                        if previous is None
                        else previous
                    )
                    manifest[full_name] = (rows, columns)
                elif local_name.startswith(cls._PT_RECURRENT_WEIGHT_PREFIXES):
                    if parameter.dim() != 2:
                        raise ValueError(f"{full_name} must be a matrix")
                    manifest[full_name] = (rows, selected)
                elif local_name.startswith(cls._PT_BIAS_PREFIXES):
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

        expected_names = {
            f"cells.{server_layer}.{local_name}"
            for server_layer in depth_layers
            for local_name, _ in cls._pt_cell(
                model=model, layer=server_layer
            ).named_parameters(recurse=False)
        } | {
            f"fc_pred.{local_name}"
            for local_name, _ in model.fc_pred.named_parameters(recurse=False)
        }
        if set(manifest) != expected_names:
            missing = sorted(expected_names - set(manifest))
            extra = sorted(set(manifest) - expected_names)
            raise NotImplementedError(
                f"recurrent ptFL coverage mismatch: missing={missing}, extra={extra}"
            )
        return PTPlan(
            manifest=manifest,
            retained_widths=retained_widths,
            is_degenerate=False,
        )

    @classmethod
    def _pt_build_plan(
        cls,
        model_name: str,
        model: nn.Module,
        capacity: float,
        selector: IndexSelector,
        depth_layers: tuple[int, ...] | None = None,
    ) -> PTPlan:
        adapter_kind = cls._pt_adapter_kind(model_name=model_name, model=model)
        if adapter_kind == "output_only":
            return cls._pt_build_output_only_plan(model=model, capacity=capacity)
        return cls._pt_build_recurrent_plan(
            model=model, capacity=capacity, selector=selector, depth_layers=depth_layers
        )

    @classmethod
    def _pt_build_model(cls, configs: Namespace) -> nn.Module:
        model_class = cls._get_objective_function(
            func_type="models", func_name=configs.model
        )
        return model_class(configs=configs)

    @classmethod
    def _pt_narrow_configs_from_state(
        cls, configs: Namespace, state: Mapping[str, torch.Tensor]
    ) -> Namespace:
        """Recover the configs of the narrow model a received submodel encodes.

        A physical submodel arrives as dense tensors with no capacity field, so
        the retained width has to be read back off the recurrent weights, whose
        first axis is the hidden width by construction.
        """

        if cls._pt_adapter_kind(model_name=configs.model) != "recurrent_stack":
            return configs
        widths = {
            int(value.shape[0])
            for name, value in state.items()
            if name.split(".")[-1].startswith(cls._PT_RECURRENT_WEIGHT_PREFIXES)
        }
        if len(widths) != 1:
            return configs
        width = widths.pop()
        if width == int(configs.hidden_size):
            return configs
        narrow_configs = copy.deepcopy(configs)
        narrow_configs.hidden_size = width
        return narrow_configs

    @classmethod
    def _pt_build_client_model(
        cls,
        configs: Namespace,
        capacity: float,
        depth_layers: tuple[int, ...] | None = None,
    ) -> nn.Module:
        adapter_kind = cls._pt_adapter_kind(model_name=configs.model)
        narrow_configs = copy.deepcopy(configs)
        if adapter_kind == "recurrent_stack":
            narrow_configs.hidden_size = cls._pt_retained_width(
                full_width=int(configs.hidden_size), capacity=capacity
            )
            if depth_layers:
                narrow_configs.num_layers = len(depth_layers)
                narrow_configs.pt_depth_layers = depth_layers
        else:
            cls._pt_validate_capacity(capacity=capacity)
        return cls._pt_build_model(configs=narrow_configs)


class ptFLLocalMetric:
    """Reports two subnet metrics beside the global-model one.

    FedProC's generalization metric is the global model on every client's test
    set. Under partial training a client does not hold the global model, and
    "the client's model" has two defensible readings that diverge whenever the
    allocation rotates:

    ``resourcelevel``
        The current global model viewed at the client's assigned width. This is
        what the papers report -- FedLAGC's "global test accuracy under a given
        client resource level", HASA's "each client is evaluated at its
        allocated width r_i". A client's width is a fixed hardware property, so
        this is stable across rounds and defined whether or not the client was
        selected.
    ``personalization``
        The subnet the client was last actually sent. This is the model it
        really holds, but for a rolling allocation like FedRolex's it is one
        arbitrary round's window, and it goes stale for a client that has not
        been selected recently.

    Their gap is how far a client's held subnet has drifted from what its width
    could deliver now. Servers whose clients hold the full global model mix this
    in nowhere, since all three metrics would coincide.

    Each one costs a full extra evaluation pass over every incumbent client, per
    dataset split, per evaluation round, so ``ptfl_local_metrics`` selects which
    to pay for. ``none`` leaves only the global metric.
    """

    _PT_LOCAL_METRICS = ("resourcelevel", "personalization")
    optional = {"ptfl_local_metrics": "personalization"}

    def _pt_init_local_metric(self) -> None:
        self._pt_local_metrics = self._pt_parse_local_metrics(
            raw=getattr(self, "ptfl_local_metrics", "personalization")
        )
        self._best_personal_loss: float = float("inf")
        # The last subnet each client was actually sent. Kept separately from
        # ``clients_personal_model_params``, which is a dict of named payloads
        # merged with ``.update()`` rather than a state dict. Only filled when
        # the metric that reads it is on, since it is a dense state dict per
        # client.
        self._pt_last_submodel: dict[int, ParameterState] = {}
        for prefix in self._pt_local_metrics:
            for dataset_type in ("train", "test"):
                self.metrics[f"{prefix}_avg_{dataset_type}_loss"] = []

    @classmethod
    def _pt_parse_local_metrics(cls, raw: str) -> tuple[str, ...]:
        names = [name.strip() for name in str(raw).split(",") if name.strip()]
        if names in ([], ["none"]):
            return ()
        unknown = sorted(set(names) - set(cls._PT_LOCAL_METRICS))
        if unknown:
            raise ValueError(
                f"unknown ptfl_local_metrics {unknown}; choose from "
                f"{list(cls._PT_LOCAL_METRICS)} or 'none'"
            )
        # Reported in a fixed order regardless of how they were listed.
        return tuple(
            prefix for prefix in cls._PT_LOCAL_METRICS if prefix in set(names)
        )

    def _pt_resource_level_params(self, client_id: int) -> ParameterState:
        """The current global model at ``client_id``'s assigned width."""

        raise NotImplementedError

    def _pt_local_map(
        self, prefix: str, incumbent: list[int]
    ) -> dict[int, ParameterState]:
        if prefix == "resourcelevel":
            return {
                client_id: self._pt_resource_level_params(client_id=client_id)
                for client_id in incumbent
            }
        # No record means the client has never been sent a subnet, so the
        # framework's fallback scores it on the global model rather than on one
        # it was never given.
        return {
            client_id: self._pt_last_submodel.get(client_id, {})
            for client_id in incumbent
        }

    def _post_eval_hook(self, dataset_type: str) -> None:
        if not self._pt_local_metrics:
            return
        incumbent = [i for i in range(self.num_clients) if not self.is_new[i]]
        for prefix in self._pt_local_metrics:
            personal_map = self._pt_local_map(prefix=prefix, incumbent=incumbent)
            losses = self.trainer.evaluate_personalized(
                ids=incumbent,
                global_params=self.public_model_params,
                personal_map=personal_map,
                dataset_type=dataset_type,
                current_iter=self.current_iter,
            )
            metric_val = float(np.mean(losses))
            self.metrics[f"{prefix}_avg_{dataset_type}_loss"].append(metric_val)
            self.logger.info(
                f"{prefix.capitalize()} {dataset_type.capitalize()} Loss: "
                f"{metric_val:.4f}"
            )
            if prefix == "personalization" and dataset_type == "test":
                self._best_personal_loss = min(self._best_personal_loss, metric_val)


class ptFL(ptFLLocalMetric, ptFLShared, tFL):
    """Base server for dense physical partial-training strategies."""

    optional = {"capacity": "1.0"}
    _pt_send_score = False

    @classmethod
    def args_update(cls, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--capacity",
            type=str,
            default=None,
            help="Comma-separated retained hidden-width ratios in (0, 1]",
        )

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        self._pt_capacities = self._parse_capacities(raw=self.capacity)
        self._pt_adapter_kind(model_name=self.configs.model, model=self.model)
        self._pt_pending_manifests: dict[int, Manifest] = {}
        # Every physical-partial client trains and deploys a narrow submodel.
        self._pt_init_local_metric()

    @classmethod
    def _parse_capacities(cls, raw: str) -> tuple[float, ...]:
        return cls._pt_parse_ratios(raw=raw, option_name="capacity")

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

    def _pt_select_depth(
        self, client_id: int, num_layers: int
    ) -> tuple[int, ...] | None:
        """Retain full depth unless a strategy overrides the selection."""

        del client_id, num_layers
        return None

    def _pt_aggregation_weight(
        self, client_id: int, package: Mapping[str, Any]
    ) -> float:
        del client_id
        return float(package["score"]) if self._pt_send_score else 1.0

    def _pt_client_plan(
        self, client_id: int
    ) -> tuple[float, tuple[int, ...] | None, PTPlan]:
        """Build one client's extraction plan without recording a manifest.

        ``package`` commits the manifest for the round; evaluation reuses the
        same plan read-only, so the two paths share this builder.
        """

        capacity = self._pt_capacity_for_client(client_id=client_id)
        num_layers = len(self.model.cells) if hasattr(self.model, "cells") else None
        depth_layers = (
            self._pt_select_depth(client_id=client_id, num_layers=num_layers)
            if num_layers is not None
            else None
        )
        plan = self._pt_build_plan(
            model_name=self.configs.model,
            model=self.model,
            capacity=capacity,
            selector=lambda group_name, full_width, retained: self._pt_select_indices(
                group_name=group_name,
                full_width=full_width,
                retained=retained,
                client_id=client_id,
            ),
            depth_layers=depth_layers,
        )
        return capacity, depth_layers, plan

    def _pt_resource_level_params(self, client_id: int) -> ParameterState:
        _capacity, _depth, plan = self._pt_client_plan(client_id=client_id)
        return self._pt_extract_parameters(
            parameters=self.public_model_params, manifest=plan.manifest
        )

    def package(self, client_id: int) -> dict[str, Any]:
        capacity, depth_layers, plan = self._pt_client_plan(client_id=client_id)
        self._pt_pending_manifests[client_id] = plan.manifest
        submodel = self._pt_extract_parameters(
            parameters=self.public_model_params, manifest=plan.manifest
        )
        if "personalization" in self._pt_local_metrics:
            # This is the client's model until it is sent another one, so it is
            # what the last-sent metric must score.
            self._pt_last_submodel[client_id] = submodel
        wire = ["regular_model_params", "capacity"]
        package = {
            "client_id": client_id,
            "current_iter": self.current_iter,
            "regular_model_params": submodel,
            "capacity": capacity,
            "scheduler_state": self.client_scheduler_states[client_id],
        }
        if depth_layers is not None:
            package["depth_layers"] = depth_layers
            wire.append("depth_layers")
        package["__wire__"] = tuple(wire)
        return package

    def _pt_accumulate_client_updates(
        self, packages: OrderedDict[int, dict[str, Any]]
    ) -> tuple[ParameterState, ParameterState, float]:
        """Validate and accumulate physical-submodel payloads once."""
        accum: ParameterState = OrderedDict()
        counts: ParameterState = OrderedDict()
        total_weight = 0.0
        for name, parameter in self.public_model_params.items():
            accum[name] = torch.zeros_like(parameter)
            counts[name] = torch.zeros_like(parameter, dtype=torch.float32)

        for client_id, package in packages.items():
            if client_id not in self._pt_pending_manifests:
                raise KeyError(
                    f"missing server-side PT manifest for client {client_id}"
                )
            manifest = self._pt_pending_manifests.pop(client_id)
            local_parameters = package["regular_model_params"]
            if set(local_parameters) != set(manifest):
                missing = sorted(set(manifest) - set(local_parameters))
                extra = sorted(set(local_parameters) - set(manifest))
                raise KeyError(
                    f"client {client_id} parameter mismatch: missing={missing}, "
                    f"extra={extra}"
                )
            weight = self._pt_aggregation_weight(
                client_id=client_id, package=package
            )
            if not math.isfinite(weight) or weight <= 0.0:
                raise ValueError("partial-update weights must be positive and finite")
            total_weight += weight

            for name, indices in manifest.items():
                server_parameter = self.public_model_params[name]
                local = local_parameters[name].to(server_parameter)
                expected_shape = tuple(index.numel() for index in indices)
                if tuple(local.shape) != expected_shape:
                    raise ValueError(
                        f"client {client_id} {name}: shape {tuple(local.shape)}; "
                        f"expected {expected_shape}"
                    )
                if not indices:
                    accum[name].add_(local, alpha=weight)
                    counts[name].add_(weight)
                    continue
                grid = tuple(torch.meshgrid(*indices, indexing="ij"))
                accum[name][grid] += local * weight
                counts[name][grid] += weight
        return accum, counts, total_weight

    def aggregate_client_updates(
        self, packages: OrderedDict[int, dict[str, Any]]
    ) -> None:
        """Average covered coordinates while preserving untouched parameters."""

        accum, counts, _total_weight = self._pt_accumulate_client_updates(
            packages=packages
        )

        new_parameters = OrderedDict()
        for name, original in self.public_model_params.items():
            count = counts[name]
            updated = accum[name] / count.clamp_min(1.0).to(accum[name].dtype)
            new_parameters[name] = torch.where(
                condition=count > 0,
                input=updated,
                other=original,
            )
        self._commit_global(new_params=new_parameters)


class ptFL_Client(ptFLShared, tFL_Client):
    """Stateless client that rebuilds its physical narrow model each round."""

    _pt_send_score = False

    def set_parameters(self, package: Mapping[str, Any]) -> None:
        self.id = int(package["client_id"])
        self.current_iter = int(package["current_iter"])
        self._load_private(client_id=self.id)

        capacity = float(package["capacity"])
        depth_layers = package.get("depth_layers")
        depth_layers = tuple(depth_layers) if depth_layers else None
        self._pt_depth_layers = depth_layers
        self.model = self._pt_build_client_model(
            configs=self.configs, capacity=capacity, depth_layers=depth_layers
        )
        state = package["regular_model_params"]
        if depth_layers is not None and isinstance(self.model.cells, nn.ModuleList):
            server_to_client = {
                server_layer: client_layer
                for client_layer, server_layer in enumerate(depth_layers)
            }
            state = self._pt_rename_state(state=state, layer_map=server_to_client)
        self.model.load_state_dict(state_dict=state, strict=True)
        self.optimizer = self._build(kind="optimizers", name=self.configs.optimizer)(
            params=self.model.parameters(),
            configs=self.configs,
        )
        self._scheduler_base_lrs = [
            float(group["lr"]) for group in self.optimizer.param_groups
        ]
        self.initialize_scheduler()
        if self.scheduler_mode == "iteration" and package.get("scheduler_state"):
            self.restore_scheduler(
                scheduler=self.scheduler,
                optimizer=self.optimizer,
                state=package["scheduler_state"],
                mode=self.scheduler_mode,
            )
        self.regular_params_name = [name for name, _ in self.model.named_parameters()]
        self.personal_params_name = []

    def package(self) -> dict[str, Any]:
        regular = OrderedDict(
            (name, parameter.detach().cpu().clone())
            for name, parameter in self.model.named_parameters()
        )
        depth_layers = self._pt_depth_layers
        if depth_layers is not None and isinstance(self.model.cells, nn.ModuleList):
            client_to_server = dict(enumerate(depth_layers))
            regular = self._pt_rename_state(state=regular, layer_map=client_to_server)
        wire = ["regular_model_params"]
        if self._pt_send_score:
            wire.append("score")
        return {
            "__wire__": tuple(wire),
            "client_id": self.id,
            "regular_model_params": regular,
            "personal_model_params": {},
            "optimizer_state": {},
            "scheduler_state": copy.deepcopy(self.scheduler.state_dict()),
            "score": self.train_samples,
        }

    def evaluate_global(
        self,
        client_id: int,
        global_params: OrderedDict[str, torch.Tensor],
        dataset_type: str,
        current_iter: int,
    ) -> float:
        self.id = client_id
        self._load_private(client_id=client_id)
        # Rebuild at the width the incoming submodel actually encodes: full for
        # the global-model metric, narrow when the server evaluates a client at
        # its assigned subnet.
        self.model = self._pt_build_model(
            configs=self._pt_narrow_configs_from_state(
                configs=self.configs, state=global_params
            )
        )
        return super().evaluate_global(
            client_id=client_id,
            global_params=global_params,
            dataset_type=dataset_type,
            current_iter=current_iter,
        )

    def evaluate_personalized(
        self,
        client_id: int,
        global_params: OrderedDict[str, torch.Tensor],
        personal_params: dict[str, torch.Tensor],
        dataset_type: str,
        current_iter: int,
    ) -> float:
        # The personalized metric is the client's own submodel, which for a
        # physical-partial strategy is narrower than the global model, so it
        # replaces the global state rather than overlaying on it.
        return self.evaluate_global(
            client_id=client_id,
            global_params=personal_params or global_params,
            dataset_type=dataset_type,
            current_iter=current_iter,
        )


class ptFLUpdateShared(ptFLShared):
    """Masked full-forward partial updates shared by servers and clients."""

    @staticmethod
    def _pt_parameter_groups(model: nn.Module) -> OrderedDict[str, tuple[str, ...]]:
        groups: OrderedDict[str, tuple[str, ...]] = OrderedDict()
        for module_name, module in model.named_modules():
            names = tuple(
                f"{module_name}.{name}" if module_name else name
                for name, _ in module.named_parameters(recurse=False)
            )
            if names:
                groups[module_name] = names
        return groups

    @classmethod
    def _pt_mask_from_spec(
        cls,
        model: nn.Module,
        spec: Mapping[str, tuple[int, ...] | None],
    ) -> ParameterState:
        groups = cls._pt_parameter_groups(model=model)
        unknown = set(spec) - set(groups)
        if unknown:
            raise KeyError(f"unknown trainable parameter groups: {sorted(unknown)}")

        parameters = dict(model.named_parameters())
        masks: ParameterState = OrderedDict()
        for module_name, parameter_names in groups.items():
            rows = spec.get(module_name, ())
            for name in parameter_names:
                parameter = parameters[name]
                mask = torch.zeros_like(parameter, dtype=torch.bool)
                if rows is None:
                    mask.fill_(True)
                elif rows:
                    if parameter.ndim == 0:
                        raise ValueError(f"{name} cannot be partitioned by output row")
                    index = torch.as_tensor(rows, dtype=torch.long)
                    if bool(((index < 0) | (index >= parameter.shape[0])).any()):
                        raise IndexError(f"{name} trainable rows exceed dimension 0")
                    if index.unique().numel() != index.numel():
                        raise ValueError(f"{name} trainable rows must be unique")
                    mask[index] = True
                masks[name] = mask
        return masks


class ptFLUpdate(ptFLUpdateShared, tFL):
    """Server base for full-forward, masked-backward partial training."""

    compulsory = {"return_diff": True}
    _pt_send_spec = True
    _pt_weighted_aggregation = False

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        self._pt_pending_update_masks: dict[int, ParameterState] = {}

    def _pt_update_spec(self, client_id: int) -> TrainableSpec:
        raise NotImplementedError("ptFLUpdate subclasses must define trainable groups")

    def package(self, client_id: int) -> dict[str, Any]:
        spec = self._pt_update_spec(client_id=client_id)
        self._pt_pending_update_masks[client_id] = self._pt_mask_from_spec(
            model=self.model, spec=spec
        )
        package = {
            "__wire__": (
                ("regular_model_params", "trainable_spec")
                if self._pt_send_spec
                else ("regular_model_params",)
            ),
            "client_id": client_id,
            "current_iter": self.current_iter,
            "regular_model_params": copy.deepcopy(self.public_model_params),
            "personal_model_params": self.clients_personal_model_params[client_id],
            "optimizer_state": self.client_optimizer_states[client_id],
            "scheduler_state": self.client_scheduler_states[client_id],
        }
        if self._pt_send_spec:
            package["trainable_spec"] = spec
        return package

    def aggregate_client_updates(
        self, packages: OrderedDict[int, dict[str, Any]]
    ) -> None:
        accum = OrderedDict(
            (name, torch.zeros_like(parameter))
            for name, parameter in self.public_model_params.items()
        )
        counts = OrderedDict(
            (name, torch.zeros_like(parameter, dtype=torch.float32))
            for name, parameter in self.public_model_params.items()
        )
        for client_id, package in packages.items():
            mask = self._pt_pending_update_masks.pop(client_id)
            differences = package["model_params_diff"]
            expected_names = {
                name for name, selected in mask.items() if bool(selected.any())
            }
            if set(differences) != expected_names:
                missing = sorted(expected_names - set(differences))
                extra = sorted(set(differences) - expected_names)
                raise KeyError(
                    f"client {client_id} update mismatch: missing={missing}, "
                    f"extra={extra}"
                )
            weight = float(package["score"]) if self._pt_weighted_aggregation else 1.0
            if not math.isfinite(weight) or weight <= 0.0:
                raise ValueError("partial-update weights must be positive and finite")
            for name, selected in differences.items():
                expected = int(mask[name].sum().item())
                if selected.numel() != expected:
                    raise ValueError(
                        f"client {client_id} {name}: {selected.numel()} values; "
                        f"expected {expected}"
                    )
                accum[name][mask[name]] += selected.to(accum[name]) * weight
                counts[name][mask[name]] += weight

        updated: ParameterState = OrderedDict()
        for name, original in self.public_model_params.items():
            count = counts[name]
            mean_difference = accum[name] / count.clamp_min(1.0).to(accum[name])
            updated[name] = torch.where(
                condition=count > 0,
                input=original - mean_difference,
                other=original,
            )
        self._commit_global(new_params=updated)


class ptFLUpdate_Client(ptFLUpdateShared, tFL_Client):
    """Client base for full-forward, masked-backward partial training."""

    return_diff = True
    return_diff_score = False
    _pt_send_score = False

    def _pt_resolve_update_spec(
        self, package: Mapping[str, Any]
    ) -> Mapping[str, tuple[int, ...] | None]:
        return package["trainable_spec"]

    def _pt_resolve_update_mask(
        self, package: Mapping[str, Any]
    ) -> ParameterState:
        return self._pt_mask_from_spec(
            model=self.model,
            spec=self._pt_resolve_update_spec(package=package),
        )

    def _pt_zero_frozen_optimizer_state(
        self, optimizer: torch.optim.Optimizer
    ) -> None:
        for parameter, mask, _initial in self._pt_frozen_parameters:
            for value in optimizer.state[parameter].values():
                if torch.is_tensor(value) and value.shape == parameter.shape:
                    value.masked_fill_(~mask.to(device=value.device), 0)

    def _pt_after_optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        del args, kwargs
        for index, (parameter, mask, initial) in enumerate(
            self._pt_frozen_parameters
        ):
            if mask.device != parameter.device:
                mask = mask.to(device=parameter.device)
                initial = initial.to(device=parameter.device)
                self._pt_frozen_parameters[index] = (parameter, mask, initial)
            with torch.no_grad():
                parameter[~mask] = initial[~mask]
        self._pt_zero_frozen_optimizer_state(optimizer=optimizer)

    def _pt_install_optimizer_guard(self) -> None:
        self._pt_frozen_parameters = [
            (parameter, self._pt_trainable_mask[name], parameter.detach().clone())
            for name, parameter in self.model.named_parameters()
            if bool(self._pt_trainable_mask[name].any())
            and not bool(self._pt_trainable_mask[name].all())
        ]
        self._pt_zero_frozen_optimizer_state(optimizer=self.optimizer)
        self._pt_optimizer_handle = self.optimizer.register_step_post_hook(
            self._pt_after_optimizer_step
        )

    def _pt_mask_gradient(self, name: str, gradient: torch.Tensor) -> torch.Tensor:
        mask = self._pt_trainable_mask[name]
        if mask.device != gradient.device:
            mask = mask.to(device=gradient.device)
            self._pt_trainable_mask[name] = mask
        return gradient * mask

    def set_parameters(self, package: dict[str, Any]) -> None:
        for handle in getattr(self, "_pt_gradient_handles", []):
            handle.remove()
        optimizer_handle = getattr(self, "_pt_optimizer_handle", None)
        if optimizer_handle is not None:
            optimizer_handle.remove()
        for parameter in self.model.parameters():
            parameter.requires_grad_(True)
        super().set_parameters(package=package)

        self._pt_trainable_mask = self._pt_resolve_update_mask(
            package=package,
        )
        self._pt_gradient_handles = []
        for name, parameter in self.model.named_parameters():
            mask = self._pt_trainable_mask[name]
            if bool(mask.all()):
                continue
            if not bool(mask.any()):
                parameter.requires_grad_(False)
                continue
            self._pt_gradient_handles.append(
                parameter.register_hook(
                    lambda gradient, name=name: self._pt_mask_gradient(
                        name=name, gradient=gradient
                    )
                )
            )
        self._pt_install_optimizer_guard()

    def package(self) -> dict[str, Any]:
        package = super().package()
        package["model_params_diff"] = OrderedDict(
            # The cached mask follows the gradient's device, which is not the
            # device the outgoing diff sits on once the payload is staged.
            (name, difference[mask.to(device=difference.device)].clone())
            for name, difference in package["model_params_diff"].items()
            if bool((mask := self._pt_trainable_mask[name]).any())
        )
        wire = ["model_params_diff"]
        if self._pt_send_score:
            wire.append("score")
        package["__wire__"] = tuple(wire)
        return package
