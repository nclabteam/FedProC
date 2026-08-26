import copy
from collections import OrderedDict
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from .tFL import tFL, tFL_Client


class FedCross(tFL):
    """FedCross multi-model cross-aggregation (Hu et al., ICDE 2024)."""

    optional = {
        "cross_alpha": 0.99,
        "collaborative_model_select_strategy": 1,
    }

    @classmethod
    def args_update(cls, parser: Any) -> None:
        parser.add_argument("--cross_alpha", type=float, default=None)
        parser.add_argument(
            "-cmss",
            "--collaborative_model_select_strategy",
            type=int,
            default=None,
            choices=[0, 1, 2],
        )

    def __init__(self, configs: Any, times: Any) -> None:
        super().__init__(configs=configs, times=times)
        if self.random_join_ratio:
            raise ValueError("FedCross requires a fixed number of clients per round")
        if not 0.5 <= self.cross_alpha < 1:
            raise ValueError("cross_alpha must be in [0.5, 1)")
        self.middleware_models = [
            copy.deepcopy(self.public_model_params)
            for _ in range(self.num_join_clients)
        ]

    def package(self, client_id: int) -> dict:
        package = super().package(client_id=client_id)
        slot = self.selected_clients.index(client_id)
        package["regular_model_params"] = copy.deepcopy(self.middleware_models[slot])
        return package

    def aggregate_client_updates(self, packages: "OrderedDict[int, dict]") -> None:
        uploaded = [
            copy.deepcopy(package["regular_model_params"])
            for package in packages.values()
        ]
        if len(uploaded) > 1:
            uploaded = self._cross_aggregation(
                models=uploaded,
                similarities=self._calculate_similarity(models=uploaded),
            )
        self.middleware_models = uploaded

        new_global = OrderedDict(
            (
                name,
                torch.stack([model[name].float() for model in uploaded])
                .mean(dim=0)
                .to(self.public_model_params[name].dtype),
            )
            for name in self.public_model_params
        )
        self._commit_global(new_params=new_global)

    @staticmethod
    def _calculate_similarity(
        models: List[Dict[str, torch.Tensor]],
    ) -> List[List[float]]:
        flattened = [
            torch.cat([parameter.float().reshape(-1) for parameter in model.values()])
            for model in models
        ]
        similarities = [[0.0] * len(models) for _ in models]
        for i in range(len(models)):
            similarities[i][i] = 1.0
            for j in range(i):
                value = F.cosine_similarity(flattened[i], flattened[j], dim=0).item()
                similarities[i][j] = value
                similarities[j][i] = value
        return similarities

    def _cross_aggregation(
        self,
        models: List[Dict[str, torch.Tensor]],
        similarities: List[List[float]],
    ) -> List[Dict[str, torch.Tensor]]:
        count = len(models)
        offset = self.current_iter % (count - 1) + 1
        result = []
        for index, model in enumerate(models):
            candidates = [peer for peer in range(count) if peer != index]
            if self.collaborative_model_select_strategy == 0:
                peer = (index + offset) % count
            elif self.collaborative_model_select_strategy == 1:
                peer = min(candidates, key=lambda item: similarities[index][item])
            else:
                peer = max(candidates, key=lambda item: similarities[index][item])
            result.append(
                self._aggregate_parameters_cross(
                    state_dicts=[model, models[peer]],
                    weights=[self.cross_alpha, 1 - self.cross_alpha],
                )
            )
        return result

    @staticmethod
    def _aggregate_parameters_cross(
        state_dicts: List[Dict[str, torch.Tensor]],
        weights: List[float],
    ) -> Dict[str, torch.Tensor]:
        return {
            name: sum(
                state[name].float() * weight
                for state, weight in zip(state_dicts, weights)
            ).to(state_dicts[0][name].dtype)
            for name in state_dicts[0]
        }


class FedCross_Client(tFL_Client):
    def package(self) -> dict:
        package = super().package()
        package["__wire__"] = ("regular_model_params",)
        return package
