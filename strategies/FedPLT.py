"""Federated Learning with Partial Layer Training."""

from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import Any, Mapping

from torch import nn

from .ptFL import TrainableSpec, ptFLUpdate, ptFLUpdate_Client


class FedPLTShared:
    """Fixed FedPLT allocation shared by the server and stateless clients."""

    @staticmethod
    def _fedplt_layer_ratios(
        parameter_counts: tuple[int, ...], training_ratio: float
    ) -> tuple[float, ...]:
        total = sum(parameter_counts)
        caps = [count / (training_ratio * total) for count in parameter_counts]
        remaining = 1.0
        active = set(range(len(caps)))
        contributions = [0.0] * len(caps)
        while active:
            tau = remaining / len(active)
            saturated = {index for index in active if caps[index] < tau}
            if not saturated:
                for index in active:
                    contributions[index] = tau
                break
            for index in saturated:
                contributions[index] = caps[index]
                remaining -= caps[index]
            active -= saturated

        # Paper projection solution: q_l = x_l * r_k * sum_j(h_j) / h_l.
        return tuple(
            contribution * training_ratio * total / count
            for contribution, count in zip(contributions, parameter_counts)
        )

    @staticmethod
    def _fedplt_rotating_blocks(
        retained_counts: tuple[int, ...], sublayers: int
    ) -> tuple[tuple[int, ...], ...]:
        cursor = 0
        assignments = []
        for retained in retained_counts:
            assignments.append(
                tuple((cursor + offset) % sublayers for offset in range(retained))
            )
            cursor = (cursor + retained) % sublayers
        return tuple(assignments)

    @classmethod
    def _fedplt_update_specs(
        cls,
        model: nn.Module,
        ratios: tuple[float, ...],
        block_size: int,
        num_clients: int,
    ) -> dict[int, TrainableSpec]:
        if len(ratios) == 1:
            ratios = ratios * num_clients
        elif len(ratios) != num_clients:
            raise ValueError("fedplt_ratios must contain one ratio or one per client")

        groups = cls._pt_parameter_groups(model=model)
        parameters = dict(model.named_parameters())
        counts = tuple(
            sum(parameters[name].numel() for name in names) for names in groups.values()
        )
        layer_rows = []
        for module_name, names in groups.items():
            dimensions = {parameters[name].shape[0] for name in names}
            if len(dimensions) != 1:
                raise ValueError(
                    f"FedPLT group {module_name!r} must share one output dimension"
                )
            rows = dimensions.pop()
            if rows % block_size:
                raise ValueError(
                    f"FedPLT group {module_name!r} has {rows} rows, not divisible "
                    f"by block size {block_size}"
                )
            layer_rows.append(rows)

        layer_ratios = tuple(
            cls._fedplt_layer_ratios(
                parameter_counts=counts,
                training_ratio=ratio,
            )
            for ratio in ratios
        )
        assignments_by_layer = []
        for layer_index, rows in enumerate(layer_rows):
            sublayers = rows // block_size
            retained_counts = tuple(
                min(sublayers, max(1, round(ratios_[layer_index] * sublayers)))
                for ratios_ in layer_ratios
            )
            assignments_by_layer.append(
                cls._fedplt_rotating_blocks(
                    retained_counts=retained_counts,
                    sublayers=sublayers,
                )
            )

        specs: dict[int, TrainableSpec] = {}
        for client_id in range(num_clients):
            spec: TrainableSpec = OrderedDict()
            for layer_index, module_name in enumerate(groups):
                spec[module_name] = tuple(
                    row
                    for sublayer in assignments_by_layer[layer_index][client_id]
                    for row in range(
                        sublayer * block_size,
                        (sublayer + 1) * block_size,
                    )
                )
            specs[client_id] = spec
        return specs


class FedPLT(FedPLTShared, ptFLUpdate):
    """Train fixed rotational sublayers and aggregate their sparse updates."""

    optional = {
        "fedplt_ratios": "0.29",
        "fedplt_block_size": 1,
    }
    _pt_send_spec = False
    _pt_weighted_aggregation = True

    @classmethod
    def args_update(cls, parser: ArgumentParser) -> None:
        parser.add_argument("--fedplt_ratios", type=str, default=None)
        parser.add_argument("--fedplt_block_size", type=int, default=None)

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        self._fedplt_ratios = self._pt_parse_ratios(
            raw=self.fedplt_ratios, option_name="fedplt_ratios"
        )
        if self.fedplt_block_size < 1:
            raise ValueError("fedplt_block_size must be positive")
        self._fedplt_specs = self._fedplt_update_specs(
            model=self.model,
            ratios=self._fedplt_ratios,
            block_size=self.fedplt_block_size,
            num_clients=self.num_clients,
        )

    def _pt_update_spec(self, client_id: int) -> TrainableSpec:
        return self._fedplt_specs[client_id]


class FedPLT_Client(FedPLTShared, ptFLUpdate_Client):
    """FedPLT client returning only trained sublayer updates."""

    _pt_send_score = True

    def __init__(self, configs: Namespace, times: int, device: str) -> None:
        super().__init__(configs=configs, times=times, device=device)
        self._fedplt_ratios = self._pt_parse_ratios(
            raw=self.fedplt_ratios, option_name="fedplt_ratios"
        )
        if self.fedplt_block_size < 1:
            raise ValueError("fedplt_block_size must be positive")
        self._fedplt_specs = self._fedplt_update_specs(
            model=self.model,
            ratios=self._fedplt_ratios,
            block_size=self.fedplt_block_size,
            num_clients=self.num_clients,
        )

    def _pt_resolve_update_spec(self, package: Mapping[str, Any]) -> TrainableSpec:
        return self._fedplt_specs[self.id]
