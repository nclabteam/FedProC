from collections import OrderedDict

import torch

from ._core import StatelessClient, StatelessPFLServer


class FedBN(StatelessPFLServer):
    """
    FedBN: Federated Learning on Non-IID Features via Local Batch Normalization.

    Identical to FedAvg except Batch Normalization parameters (matched by 'bn'
    in the name) are excluded from aggregation and kept local (personal). For
    models without BN layers this degenerates to standard FedAvg.

    Reference: Li et al., ICLR 2021. arXiv 2102.07623.
    """

    def aggregate_client_updates(self, packages):
        scores = [p["score"] for p in packages.values()]
        total = float(sum(scores))
        weights = torch.tensor([s / total for s in scores], dtype=torch.float32)
        new_params = OrderedDict()
        for name in self.public_model_params:
            if "bn" in name.lower():
                new_params[name] = self.public_model_params[name]
                continue
            stacked = torch.stack(
                [p["regular_model_params"][name] for p in packages.values()], dim=-1
            )
            new_params[name] = torch.sum(stacked * weights.to(stacked.dtype), dim=-1)
        self._commit_global(new_params)


class FedBN_Client(StatelessClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.personal_params_name = [
            name for name in self.regular_params_name if "bn" in name.lower()
        ]
