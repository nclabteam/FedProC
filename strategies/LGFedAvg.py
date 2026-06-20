from collections import OrderedDict
from typing import List, Set

import torch

from .tFL import tFL, tFL_Client


def _global_param_names(names: List[str], num_global_layers: int) -> Set[str]:
    """Names belonging to the first ``num_global_layers`` top-level module groups."""
    seen: list = []
    for name in names:
        prefix = name.split(".")[0]
        if prefix not in seen:
            seen.append(prefix)
    global_prefixes = set(seen[:num_global_layers])
    return {name for name in names if name.split(".")[0] in global_prefixes}


class LGFedAvg(tFL):
    """
    LG-FedAvg: shared global body (first ``num_global_layers`` module groups)
    aggregated across clients; personal head stays local. Matches the legacy
    behaviour where non-global params are zeroed in the global model.

    Reference: Liang et al., arXiv 2001.01523.
    """

    optional = {
        "num_global_layers": 1,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--num_global_layers", type=int, default=None)

    def aggregate_client_updates(self, packages):
        global_names = _global_param_names(
            list(self.public_model_params.keys()), self.num_global_layers
        )
        scores = [p["score"] for p in packages.values()]
        total = float(sum(scores))
        weights = torch.tensor([s / total for s in scores], dtype=torch.float32)
        new_params = OrderedDict()
        for name in self.public_model_params:
            if name not in global_names:
                new_params[name] = torch.zeros_like(self.public_model_params[name])
                continue
            stacked = torch.stack(
                [p["regular_model_params"][name] for p in packages.values()], dim=-1
            )
            new_params[name] = torch.sum(stacked * weights.to(stacked.dtype), dim=-1)
        self._commit_global(new_params)


class LGFedAvg_Client(tFL_Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        global_names = _global_param_names(
            self.regular_params_name, self.num_global_layers
        )
        self.personal_params_name = [
            name for name in self.regular_params_name if name not in global_names
        ]
