from collections import OrderedDict

from ._core import StatelessServer


class FedAvgM(StatelessServer):

    optional = {"server_momentum": 0.9, "server_learning_rate": 0.01}

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--server_momentum", type=float, default=None)
        parser.add_argument("--server_learning_rate", type=float, default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum_vector = None

    def aggregate_client_updates(self, packages):
        prev = OrderedDict(
            (k, v.clone()) for k, v in self.public_model_params.items()
        )
        super().aggregate_client_updates(packages)  # FedAvg step
        if self.server_momentum <= 0.0:
            return

        pseudo_gradient = {
            k: prev[k] - self.public_model_params[k] for k in prev
        }
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
        self._commit_global(new_params)
