from argparse import Namespace
from collections import OrderedDict
from typing import Any

import torch

from .pFL import pFL, pFL_Client


class FedBNShared:
    @staticmethod
    def batch_norm_state_names(model: torch.nn.Module) -> list[str]:
        names = []
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                prefix = f"{module_name}." if module_name else ""
                names.extend(prefix + name for name in module.state_dict())
        return names


class FedBN(FedBNShared, pFL):
    """FedBN: Federated Learning on Non-IID Features via Local Batch Normalization."""

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        self.bn_state_names = set(self.batch_norm_state_names(model=self.model))
        state = OrderedDict(
            (name, value.detach().cpu().clone())
            for name, value in self.model.state_dict().items()
        )
        self.public_model_params = state
        for personal in self.clients_personal_model_params.values():
            personal.update({name: state[name].clone() for name in self.bn_state_names})

    def select_clients(self) -> None:
        self._select_all_clients()

    def package(self, client_id: int) -> dict:
        package = super().package(client_id=client_id)
        package["regular_model_params"] = OrderedDict(
            (name, value)
            for name, value in package["regular_model_params"].items()
            if name not in self.bn_state_names
        )
        return package

    def aggregate_client_updates(self, packages: Any) -> None:
        new_params = OrderedDict(self.public_model_params)
        new_params.update(
            self.mean_models(
                models=[
                    package["regular_model_params"] for package in packages.values()
                ]
            )
        )
        self._commit_global(new_params=new_params)


class FedBN_Client(FedBNShared, pFL_Client):
    def __init__(self, configs: Namespace, times: int, device: str) -> None:
        super().__init__(configs=configs, times=times, device=device)
        personal = set(self.batch_norm_state_names(model=self.model))
        self.personal_params_name = [
            name for name in self.model.state_dict() if name in personal
        ]
        self.regular_params_name = [
            name for name in self.model.state_dict() if name not in personal
        ]
