from argparse import Namespace
from collections import OrderedDict
from typing import Any, List, Set

import torch

from .pFL import pFL, pFL_Client


class LGFedAvgShared:
    """Model partition shared by the LG-FedAvg server and client."""

    @staticmethod
    def global_param_names(names: List[str], num_global_layers: int) -> Set[str]:
        """Return parameters in the final top-level model groups."""
        groups = list(dict.fromkeys(name.split(".")[0] for name in names))
        if not 1 <= num_global_layers <= len(groups):
            raise ValueError(
                "num_global_layers must be between 1 and the number of "
                f"top-level model groups ({len(groups)})"
            )
        global_groups = set(groups[-num_global_layers:])
        return {name for name in names if name.split(".")[0] in global_groups}


class LGFedAvg(LGFedAvgShared, pFL):
    """Local encoders with a sample-weighted shared global predictor."""

    optional = {"num_global_layers": 1}

    @classmethod
    def args_update(cls, parser: Any) -> None:
        parser.add_argument("--num_global_layers", type=int, default=None)

    def __init__(self, configs: Any, times: Any) -> None:
        super().__init__(configs=configs, times=times)
        names = list(self.public_model_params)
        self.global_params_name = self.global_param_names(
            names=names, num_global_layers=self.num_global_layers
        )
        self.local_params_name = [
            name for name in names if name not in self.global_params_name
        ]
        for client_params in self.clients_personal_model_params.values():
            client_params.update(
                {
                    name: self.public_model_params[name].detach().cpu().clone()
                    for name in self.local_params_name
                }
            )

    def package(self, client_id: int) -> dict:
        package = super().package(client_id=client_id)
        package["regular_model_params"] = OrderedDict(
            (name, self.public_model_params[name].detach().cpu().clone())
            for name in self.public_model_params
            if name in self.global_params_name
        )
        package["__wire__"] = ("regular_model_params",)
        return package

    def aggregate_client_updates(self, packages: Any) -> None:
        client_models, scores = self.extract_models_and_scores(packages=packages)
        total = sum(scores)
        if total <= 0:
            raise ValueError("LG-FedAvg client scores must sum to a positive value")
        weights = torch.tensor([score / total for score in scores], dtype=torch.float32)
        new_params = OrderedDict()
        for name, current in self.public_model_params.items():
            if name not in self.global_params_name:
                new_params[name] = torch.zeros_like(current)
                continue
            stacked = torch.stack(
                [model[name] for model in client_models],
                dim=-1,
            )
            new_params[name] = torch.sum(stacked * weights.to(stacked.dtype), dim=-1)
        self._commit_global(new_params=new_params)


class LGFedAvg_Client(LGFedAvgShared, pFL_Client):
    def __init__(self, configs: Namespace, times: int, device: str) -> None:
        super().__init__(configs=configs, times=times, device=device)
        all_names = self.regular_params_name
        global_names = self.global_param_names(
            names=all_names, num_global_layers=self.num_global_layers
        )
        self.regular_params_name = [name for name in all_names if name in global_names]
        self.personal_params_name = [
            name for name in all_names if name not in global_names
        ]

    def package(self) -> dict:
        package = super().package()
        package["__wire__"] = ("regular_model_params",)
        return package
