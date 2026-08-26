import copy
import json
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from layers import LoRALinear

from .peftFL import peftFL, peftFL_Client


class FlexLoRAShared:
    """Rank parsing, LoRA resizing, and SVD factor redistribution."""

    @staticmethod
    def parse_client_ranks(
        value: str | Mapping[int | str, int] | None,
        num_clients: int,
        default_rank: int,
    ) -> dict[int, int]:
        parsed = json.loads(value) if isinstance(value, str) else value or {}
        if not isinstance(parsed, Mapping):
            raise ValueError("client_ranks must be a JSON object")
        ranks = {
            client_id: int(
                parsed.get(client_id, parsed.get(str(client_id), default_rank))
            )
            for client_id in range(num_clients)
        }
        if any(rank <= 0 for rank in ranks.values()):
            raise ValueError("all FlexLoRA ranks must be positive")
        return ranks

    @staticmethod
    def resize_lora_rank(model: nn.Module, rank: int) -> None:
        for module in model.modules():
            if not isinstance(module, LoRALinear) or module.r == rank:
                continue
            device, dtype = module.lora_A.device, module.lora_A.dtype
            in_features = module.lora_A.shape[0]
            out_features = module.lora_B.shape[1]
            module.r = rank
            module.scaling = module.lora_alpha / rank
            module.lora_A = nn.Parameter(
                torch.empty(in_features, rank, device=device, dtype=dtype)
            )
            module.lora_B = nn.Parameter(
                torch.zeros(rank, out_features, device=device, dtype=dtype)
            )
            nn.init.kaiming_uniform_(module.lora_A, a=5**0.5)

    @staticmethod
    def factors_from_svd(
        u: torch.Tensor,
        singular_values: torch.Tensor,
        vh: torch.Tensor,
        rank: int,
        alpha: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scaling = alpha / rank if alpha else 1.0
        a_factor = u[:, :rank]
        b_factor = (singular_values[:rank].unsqueeze(1) * vh[:rank, :]) / scaling
        return a_factor, b_factor


class FlexLoRA(FlexLoRAShared, peftFL):
    """FlexLoRA: aggregate full LoRA updates, then redistribute rank-r SVDs."""

    optional = {"client_ranks": None}

    @classmethod
    def args_update(cls, parser: ArgumentParser) -> None:
        super().args_update(parser=parser)
        parser.add_argument(
            "--client_ranks",
            default=None,
            type=str,
            help='JSON client-to-rank map, for example: {"0": 4, "1": 8}',
        )

    def __init__(self, configs: Namespace, times: int) -> None:
        ranks = self.parse_client_ranks(
            value=configs.client_ranks,
            num_clients=configs.num_clients,
            default_rank=configs.lora_r,
        )
        configs.lora_r = max(ranks.values())
        super().__init__(configs=configs, times=times)
        max_supported = min(
            min(module.lora_A.shape[0], module.lora_B.shape[1])
            for module in self.model.modules()
            if isinstance(module, LoRALinear)
        )
        if max(ranks.values()) > max_supported:
            raise ValueError(
                f"FlexLoRA rank cannot exceed the smallest layer dimension "
                f"({max_supported})"
            )
        self.client_ranks = ranks
        self.tailored_lora_params = self._initial_tailored_params()

    def _initial_tailored_params(
        self,
    ) -> dict[int, OrderedDict[str, torch.Tensor]]:
        params = self.lora_params(params=self.public_model_params)
        layers = self.lora_layers(params=params)
        tailored = {client_id: OrderedDict() for client_id in self.client_ranks}
        for names in layers.values():
            if set(names) != {"A", "B"}:
                continue
            a_name, b_name = names["A"], names["B"]
            for client_id, rank in self.client_ranks.items():
                tailored[client_id][a_name] = params[a_name][:, :rank].clone()
                tailored[client_id][b_name] = params[b_name][:rank, :].clone()
        return tailored

    def package(self, client_id: int) -> dict[str, Any]:
        package = super().package(client_id=client_id)
        package["client_rank"] = self.client_ranks[client_id]
        package["lora_model_params"] = OrderedDict(
            (name, value.clone())
            for name, value in self.tailored_lora_params[client_id].items()
        )
        package["__wire__"] = ("lora_model_params",)
        return package

    def aggregate_client_updates(self, packages: Mapping[int, dict[str, Any]]) -> None:
        items = list(packages.items())
        scores = torch.as_tensor(
            [package["score"] for _, package in items], dtype=torch.float32
        )
        if not torch.isfinite(scores).all() or scores.sum() <= 0:
            raise ValueError("FlexLoRA client scores must have a positive sum")
        weights = scores / scores.sum()
        client_params = [package["lora_model_params"] for _, package in items]
        layers = self.lora_layers(params=client_params[0])
        device = next(self.model.parameters()).device
        tailored = {client_id: OrderedDict() for client_id in self.client_ranks}
        server_params = OrderedDict()

        for names in layers.values():
            if set(names) != {"A", "B"}:
                continue
            a_name, b_name = names["A"], names["B"]
            updates = torch.stack(
                [
                    params[a_name].float()
                    @ params[b_name].float()
                    * (
                        float(self.lora_alpha) / params[a_name].shape[1]
                        if self.lora_alpha
                        else 1.0
                    )
                    for params in client_params
                ]
            ).to(device)
            # Paper Algorithm 2: W_g = sum_i gamma_i s_i B_i A_i.
            global_update = torch.tensordot(
                weights.to(device), updates, dims=([0], [0])
            )
            u, singular_values, vh = torch.linalg.svd(
                global_update, full_matrices=False
            )
            for client_id, rank in self.client_ranks.items():
                a_factor, b_factor = self.factors_from_svd(
                    u=u,
                    singular_values=singular_values,
                    vh=vh,
                    rank=rank,
                    alpha=float(self.lora_alpha),
                )
                tailored[client_id][a_name] = a_factor.detach().cpu()
                tailored[client_id][b_name] = b_factor.detach().cpu()
            server_a, server_b = self.factors_from_svd(
                u=u,
                singular_values=singular_values,
                vh=vh,
                rank=self.lora_r,
                alpha=float(self.lora_alpha),
            )
            server_params[a_name] = server_a.detach().cpu()
            server_params[b_name] = server_b.detach().cpu()

        self.tailored_lora_params = tailored
        self.update_lora_params(model=self.model, params=server_params)
        self._commit_global(
            new_params=OrderedDict(
                (name, value.detach().cpu().clone())
                for name, value in self.model.named_parameters()
            )
        )


class FlexLoRA_Client(FlexLoRAShared, peftFL_Client):
    """FlexLoRA worker whose physical adapter rank follows its client budget."""

    def __init__(self, configs: Namespace, times: int, device: str) -> None:
        super().__init__(configs=configs, times=times, device=device)
        self.client_rank = configs.lora_r

    def _rebuild_optimizer(self) -> None:
        self.optimizer = self._build(kind="optimizers", name=self.configs.optimizer)(
            params=self.model.parameters(), configs=self.configs
        )
        self._scheduler_base_lrs = [
            float(group["lr"]) for group in self.optimizer.param_groups
        ]
        self.initialize_scheduler()
        self.init_optimizer_state = copy.deepcopy(self.optimizer.state_dict())

    def set_parameters(self, package: dict[str, Any]) -> None:
        rank = int(package["client_rank"])
        if rank != self.client_rank:
            self.resize_lora_rank(model=self.model, rank=rank)
            self.client_rank = rank
            self.regular_params_name = [
                name
                for name, _ in self.model.named_parameters()
                if name.endswith(self.shared_lora_suffixes)
            ]
            self._rebuild_optimizer()

        local_package = dict(package)
        local_package["regular_model_params"] = OrderedDict(
            (name, value)
            for name, value in package["regular_model_params"].items()
            if not name.endswith((".lora_A", ".lora_B"))
        )
        super().set_parameters(package=local_package)
