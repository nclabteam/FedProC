from collections import OrderedDict

import torch

from .tFL import tFL, tFL_Client


class FedMedian(tFL):
    """Coordinate-wise median aggregation (Byzantine-robust).

    Aggregates client updates by taking the element-wise median across all
    clients, rather than a weighted mean. Tolerates up to ⌊(n-1)/2⌋ Byzantine
    workers where n is the number of clients.

    Reference: Yin et al., "Byzantine-Robust Distributed Learning: Towards
    Optimal Statistical Rates", ICML 2018.
    """

    def aggregate_client_updates(self, packages):
        new_params = OrderedDict()
        for name in self.public_model_params:
            stacked = torch.stack(
                [p["regular_model_params"][name] for p in packages.values()]
            )
            new_params[name] = torch.median(stacked, dim=0).values.clone()
        self._commit_global(new_params)


class FedMedian_Client(tFL_Client):
    pass
