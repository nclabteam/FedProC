"""FedOBD: opportunistic block dropout with adaptive deterministic quantization."""

from __future__ import annotations

import math
from argparse import Namespace
from collections import OrderedDict
from typing import Any, Mapping

import torch
from torch import nn

from .base import SharedMethods
from .tFL import tFL, tFL_Client

# Number of bits in a floating-point representation (``REPR`` in the paper).
REPR = 32
_CONTAINERS = (nn.ModuleList, nn.ModuleDict, nn.Sequential)


class FedOBDShared(SharedMethods):
    """Block decomposition and NNADQ rules shared by the server and client."""

    @classmethod
    def _obd_blocks(cls, model: nn.Module) -> "OrderedDict[str, tuple[str, ...]]":
        """Decompose a model into blocks of consecutive parameterized layers.

        Repeated structural units (the children of a ``ModuleList``,
        ``ModuleDict`` or ``Sequential``) are the paper's building blocks and
        are transmitted or dropped whole; every remaining parameterized layer
        becomes a singleton block.
        """

        blocks: OrderedDict[str, tuple[str, ...]] = OrderedDict()

        def qualify(prefix: str, name: str) -> str:
            return f"{prefix}.{name}" if prefix else name

        def visit(prefix: str, module: nn.Module) -> None:
            if isinstance(module, _CONTAINERS):
                for name, child in module.named_children():
                    visit(qualify(prefix=prefix, name=name), child)
                return
            names = tuple(
                qualify(prefix=prefix, name=name)
                for name, _ in module.named_parameters(recurse=True)
            )
            if not names:
                return
            if not any(
                isinstance(child, _CONTAINERS) for child in module.modules()
            ):
                blocks[prefix] = names
                return
            own = tuple(
                qualify(prefix=prefix, name=name)
                for name, _ in module.named_parameters(recurse=False)
            )
            if own:
                blocks[prefix] = own
            for name, child in module.named_children():
                visit(qualify(prefix=prefix, name=name), child)

        visit("", model)
        if not blocks:
            raise ValueError("FedOBD needs a model with at least one parameter")
        return blocks

    @staticmethod
    def _obd_mean_block_difference(
        previous: Mapping[str, torch.Tensor],
        current: Mapping[str, torch.Tensor],
        names: tuple[str, ...],
    ) -> float:
        """Return ``MBD(b) = ||asVect(b_prev) - asVect(b_now)|| / |b_prev|``."""

        difference = torch.cat(
            [
                (current[name].detach() - previous[name].detach())
                .to(dtype=torch.float64)
                .reshape(-1)
                for name in names
            ]
        )
        return float(difference.norm().item() / difference.numel())

    @classmethod
    def _obd_retained_blocks(
        cls,
        blocks: Mapping[str, tuple[str, ...]],
        previous: Mapping[str, torch.Tensor],
        current: Mapping[str, torch.Tensor],
        dropout_rate: float,
    ) -> tuple[str, ...]:
        """Greedily retain the highest-MBD blocks within the size budget."""

        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout rate must be in [0, 1], got {dropout_rate}")
        sizes = {
            name: sum(current[tensor].numel() for tensor in names)
            for name, names in blocks.items()
        }
        budget = (1.0 - dropout_rate) * sum(sizes.values())
        ranked = sorted(
            blocks,
            key=lambda name: (
                -cls._obd_mean_block_difference(
                    previous=previous, current=current, names=blocks[name]
                ),
                name,
            ),
        )
        revised_size = 0
        retained: list[str] = []
        for name in ranked:
            new_size = revised_size + sizes[name]
            if new_size > budget:
                continue
            revised_size = new_size
            retained.append(name)
        return tuple(retained)

    @staticmethod
    def _adq(tensor: torch.Tensor, weight: float) -> tuple[torch.Tensor, int]:
        """Return the ADQ reconstruction of one tensor and its level count.

        ``offset`` translates the vector to the minimum infinity norm,
        ``d = ||v'||_inf`` is the near-optimal normalizer, and
        ``s = floor(max(sqrt(ln4 * REPR / weight * d), 1))`` solves the
        paper's compression/informativeness trade-off.
        """

        if weight <= 0.0:
            raise ValueError(f"quantization weight must be positive, got {weight}")
        values = tensor.detach().to(dtype=torch.float64).reshape(-1)
        offset = -(values.max() + values.min()) / 2
        shifted = values + offset
        normalizer = shifted.abs().max()
        levels = int(
            math.floor(
                max(
                    math.sqrt(math.log(4.0) * REPR / weight * float(normalizer.item())),
                    1.0,
                )
            )
        )
        if float(normalizer.item()) == 0.0:
            # A constant tensor is carried exactly by its offset alone.
            return torch.full_like(tensor, float(-offset.item())), levels
        # round(v', s, d): the nearer of l/s and (l+1)/s to |v'| / d.
        quantized = torch.floor(shifted.abs() / normalizer * levels + 0.5)
        reconstructed = (
            normalizer * torch.sign(shifted) * quantized / levels
        ) - offset
        return reconstructed.reshape(tensor.shape).to(dtype=tensor.dtype), levels

    @classmethod
    def _nnadq(
        cls, tensors: Mapping[str, torch.Tensor], weight: float
    ) -> tuple["OrderedDict[str, torch.Tensor]", float]:
        """Quantize layer-structured tensors and return them with the wire size."""

        result: OrderedDict[str, torch.Tensor] = OrderedDict()
        total_bits = 0.0
        for name, tensor in tensors.items():
            reconstructed, levels = cls._adq(tensor=tensor, weight=weight)
            result[name] = reconstructed
            magnitude_bits = math.ceil(math.log2(levels + 1))
            # One sign bit per element plus the offset, normalizer and level
            # count needed to dequantize the tensor.
            total_bits += tensor.numel() * (magnitude_bits + 1) + 3 * REPR
        return result, total_bits / 8 / (1024**2)


class FedOBD(FedOBDShared, tFL):
    """Two-stage server for opportunistic block dropout."""

    optional = {
        "fedobd_dropout_rate": 0.3,
        "fedobd_weight": 0.001,
        "fedobd_epochs": 5,
        "fedobd_stage2_epochs": 10,
    }

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        # The paper runs ``R`` stage-1 rounds and *then* one stage-2 round whose
        # ``E2`` local epochs each trigger an aggregation, so stage 1 keeps the
        # full configured budget and stage 2 is charged on top of it rather than
        # carved out of it.
        self._obd_stage1_rounds = int(self.iterations)
        if self._obd_stage1_rounds < 1:
            raise ValueError(
                f"iterations must be positive; got iterations={self.iterations}"
            )
        if int(self.fedobd_stage2_epochs) < 1:
            raise ValueError("fedobd_stage2_epochs must be positive")
        self.iterations = self._obd_stage1_rounds + int(self.fedobd_stage2_epochs)
        if int(self.fedobd_epochs) < 1:
            raise ValueError("fedobd_epochs must be positive")
        self._obd_blocks_map = self._obd_blocks(model=self.model)
        self._obd_broadcast: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._obd_broadcast_iter: int | None = None
        self._obd_downlink_mb = 0.0

    def _obd_stage(self) -> int:
        return 1 if self.current_iter < self._obd_stage1_rounds else 2

    def select_clients(self) -> None:
        """Stage 1 samples a client subset; stage 2 aggregates every epoch."""

        if self._obd_stage() == 2:
            self._select_all_clients()
            return
        super().select_clients()

    def package(self, client_id: int) -> dict[str, Any]:
        if self._obd_broadcast_iter != self.current_iter:
            self._obd_broadcast, self._obd_downlink_mb = self._nnadq(
                tensors=self.public_model_params,
                weight=float(self.fedobd_weight),
            )
            self._obd_broadcast_iter = self.current_iter
        stage = self._obd_stage()
        return {
            "__wire__": ("regular_model_params",),
            "client_id": client_id,
            "current_iter": self.current_iter,
            "regular_model_params": OrderedDict(
                (name, tensor.clone())
                for name, tensor in self._obd_broadcast.items()
            ),
            "personal_model_params": self.clients_personal_model_params[client_id],
            "optimizer_state": self.client_optimizer_states[client_id],
            "scheduler_state": self.client_scheduler_states[client_id],
            "fedobd_epochs": (
                int(self.fedobd_epochs) if stage == 1 else 1
            ),
            "fedobd_dropout_rate": (
                float(self.fedobd_dropout_rate) if stage == 1 else 0.0
            ),
            "fedobd_weight": float(self.fedobd_weight),
        }

    def _compute_send_mb(
        self, packages: Mapping[int, dict[str, Any]]
    ) -> tuple[dict[int, float], float]:
        uplink = {
            client_id: float(package["fedobd_uplink_mb"])
            for client_id, package in packages.items()
        }
        return uplink, self._obd_downlink_mb * len(self.selected_clients)

    def aggregate_client_updates(
        self, packages: "OrderedDict[int, dict[str, Any]]"
    ) -> None:
        """Reconstruct each local model from its retained blocks, then average."""

        models: list[dict[str, torch.Tensor]] = []
        scores: list[float] = []
        for client_id, package in packages.items():
            reconstructed = OrderedDict(
                (name, tensor.clone())
                for name, tensor in self._obd_broadcast.items()
            )
            for name in package["fedobd_retained"]:
                if name not in self._obd_blocks_map:
                    raise KeyError(f"client {client_id} returned unknown block {name}")
                for tensor_name in self._obd_blocks_map[name]:
                    difference = package["fedobd_update"][tensor_name]
                    reconstructed[tensor_name] = reconstructed[tensor_name] + (
                        difference.to(reconstructed[tensor_name])
                    )
            models.append(reconstructed)
            scores.append(float(package["score"]))
        self._commit_global(new_params=self.mean_models(models=models, weights=scores))


class FedOBD_Client(FedOBDShared, tFL_Client):
    """Client that uploads quantized differences of its most important blocks."""

    def set_parameters(self, package: dict[str, Any]) -> None:
        super().set_parameters(package=package)
        self.epochs = int(package["fedobd_epochs"])
        self._obd_dropout_rate = float(package["fedobd_dropout_rate"])
        self._obd_weight = float(package["fedobd_weight"])
        self._obd_blocks_map = self._obd_blocks(model=self.model)
        self._obd_previous = OrderedDict(
            (name, tensor.detach().cpu().clone())
            for name, tensor in package["regular_model_params"].items()
        )

    def package(self) -> dict[str, Any]:
        result = super().package()
        current = result["regular_model_params"]
        retained = self._obd_retained_blocks(
            blocks=self._obd_blocks_map,
            previous=self._obd_previous,
            current=current,
            dropout_rate=self._obd_dropout_rate,
        )
        difference = OrderedDict(
            (name, current[name].to(dtype=torch.float32) - self._obd_previous[name])
            for block in retained
            for name in self._obd_blocks_map[block]
        )
        update, uplink_mb = self._nnadq(
            tensors=difference, weight=self._obd_weight
        )
        result["regular_model_params"] = {}
        result["fedobd_retained"] = retained
        result["fedobd_update"] = update
        # The retained block names travel with the update and identify which
        # coordinates the server may reconstruct, so they are real payload.
        result["fedobd_uplink_mb"] = uplink_mb + len(retained) * 4 / (1024**2)
        result["__wire__"] = ("fedobd_update", "fedobd_retained")
        return result
