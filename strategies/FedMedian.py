from collections import OrderedDict

import torch

from ._core import StatelessClient, StatelessServer


class FedMedian(StatelessServer):
    """Coordinate-wise median aggregation (Byzantine-robust)."""

    def aggregate_client_updates(self, packages):
        new_params = OrderedDict()
        for name in self.public_model_params:
            stacked = torch.stack(
                [p["regular_model_params"][name] for p in packages.values()]
            )
            new_params[name] = torch.median(stacked, dim=0).values.clone()
        self._commit_global(new_params)


class FedMedian_Client(StatelessClient):
    pass
