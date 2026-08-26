from argparse import Namespace
from collections import OrderedDict
from typing import Any

from .tFL import tFL


class FedAvgM(tFL):
    """FedAvgM: FedAvg with server-side SGD momentum (Hsieh et al., 2020)."""

    optional = {"server_momentum": 0.9, "server_learning_rate": 1.0}

    @classmethod
    def args_update(cls, parser: Any) -> None:
        parser.add_argument("--server_momentum", type=float, default=None)
        parser.add_argument("--server_learning_rate", type=float, default=None)

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        self.momentum_vector = None

    def aggregate_client_updates(self, packages: Any) -> None:
        prev = OrderedDict((k, v.clone()) for k, v in self.public_model_params.items())
        super().aggregate_client_updates(packages=packages)  # FedAvg step

        pseudo_gradient = {k: prev[k] - self.public_model_params[k] for k in prev}
        if self.momentum_vector is None:
            self.momentum_vector = pseudo_gradient
        else:
            self.momentum_vector = {
                k: self.server_momentum * self.momentum_vector[k] + pseudo_gradient[k]
                for k in pseudo_gradient
            }
        new_params = OrderedDict(
            (k, prev[k] - self.server_learning_rate * self.momentum_vector[k])
            for k in prev
        )
        self._commit_global(new_params=new_params)
