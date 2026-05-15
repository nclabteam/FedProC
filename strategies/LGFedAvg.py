from argparse import Namespace
from typing import Any, Dict, List, Set

from .tFL import tFL, tFL_Client


def _global_param_names(model, num_global_layers: int) -> Set[str]:
    """Return the set of parameter names that belong to the first `num_global_layers`
    unique top-level module groups (identified by the first dot-separated prefix)."""
    # Collect unique top-level layer names in order of first appearance
    seen_layers: list = []
    for name, _ in model.named_parameters():
        prefix = name.split(".")[0]
        if prefix not in seen_layers:
            seen_layers.append(prefix)

    global_prefixes = set(seen_layers[:num_global_layers])

    return {
        name
        for name, _ in model.named_parameters()
        if name.split(".")[0] in global_prefixes
    }


class LGFedAvg(tFL):
    """
    LG-FedAvg: Think Locally, Act Globally.

    Splits the model into a shared global body (first `num_global_layers`
    top-level module groups) and a personal head (remaining layers).
    Only the global body is aggregated across clients; the personal head
    stays on the device and is never sent to the server.

    Reference: Liang et al., "Think Locally, Act Globally: Federated Learning
    with Local and Global Representations", ArXiv 2020. arXiv 2001.01523.
    """

    optional = {
        "num_global_layers": 1,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--num_global_layers", type=int, default=None)

    def _global_names(self) -> Set[str]:
        return _global_param_names(self.model, self.num_global_layers)

    def aggregate_models(self) -> None:
        global_names = self._global_names()
        self.model = self.reset_model(self.model)
        for client, weight in zip(self.client_data, self.weights):
            for (name, global_param), local_param in zip(
                self.model.named_parameters(),
                client["model"].parameters(),
            ):
                if name not in global_names:
                    continue
                global_param.data.add_(
                    local_param.data.to(global_param.device), alpha=weight
                )


class LGFedAvg_Client(tFL_Client):
    """
    Client for LG-FedAvg. Receives the updated global layers from the server
    and applies them, while keeping personal layers unchanged.
    """

    def __init__(self, configs: Namespace, id: int, times: int) -> None:
        super().__init__(configs=configs, id=id, times=times)
        self._global_names: Set[str] = set()

    def receive_from_server(self, data: dict) -> None:
        global_names = _global_param_names(data["model"], self.num_global_layers)
        self._global_names = global_names
        for (name, old_param), new_param in zip(
            self.model.named_parameters(),
            data["model"].parameters(),
        ):
            if name in global_names:
                old_param.data.copy_(new_param.data)
