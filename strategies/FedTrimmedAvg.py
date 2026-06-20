from collections import OrderedDict

import torch

from .tFL import tFL, tFL_Client


class FedTrimmedAvg(tFL):
    """Coordinate-wise trimmed-mean aggregation (Byzantine-robust)."""

    optional = {
        "beta": 0.2,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument(
            "--beta",
            type=float,
            default=None,
            help="Fraction to cut off of both tails of the distribution",
        )

    def aggregate_client_updates(self, packages):
        n = len(packages)
        lowercut = int(self.beta * n)
        uppercut = n - lowercut
        if lowercut > uppercut:
            raise ValueError(
                "The fraction to cut off of both tails of the distribution is too large"
            )
        new_params = OrderedDict()
        for name in self.public_model_params:
            stacked = torch.stack(
                [p["regular_model_params"][name] for p in packages.values()]
            )
            sorted_layers, _ = torch.sort(stacked, dim=0)
            new_params[name] = torch.mean(sorted_layers[lowercut:uppercut], dim=0)
        self._commit_global(new_params)


class FedTrimmedAvg_Client(tFL_Client):
    pass
