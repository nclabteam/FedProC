from collections import OrderedDict

import torch

from .tFL import tFL, tFL_Client


class FedAdam(tFL):
    """FedAdam — server-side Adam optimizer for federated learning.

    After each FedAvg aggregation round, treats the average client update
    Δ = (FedAvg result − previous global model) as a pseudo-gradient and
    applies bias-corrected Adam on the server side.

    Reference: Reddi et al., "Adaptive Federated Optimization", ICLR 2021.
    arXiv:2003.00295.
    """

    optional = {
        "beta1_server": 0.9,
        "beta2_server": 0.999,
        "eta_server": 1e-3,
        "tau_server": 1e-9,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--beta1_server", type=float, default=None)
        parser.add_argument("--beta2_server", type=float, default=None)
        parser.add_argument("--eta_server", type=float, default=None)
        parser.add_argument("--tau_server", type=float, default=None)

    def __init__(self, configs, times):
        super().__init__(configs=configs, times=times)
        self.m_t = {k: torch.zeros_like(v) for k, v in self.public_model_params.items()}
        self.v_t = {k: torch.zeros_like(v) for k, v in self.public_model_params.items()}

    def aggregate_client_updates(self, packages):
        prev = OrderedDict((k, v.clone()) for k, v in self.public_model_params.items())
        super().aggregate_client_updates(packages)  # FedAvg step → self.public_model_params

        # Δ^t = client update direction (FedAvg result − previous global params)
        delta = {k: self.public_model_params[k] - prev[k] for k in prev}

        t = self.current_iter + 1.0
        for k in prev:
            self.m_t[k] = self.beta1_server * self.m_t[k] + (1 - self.beta1_server) * delta[k]
            self.v_t[k] = self.beta2_server * self.v_t[k] + (1 - self.beta2_server) * delta[k] ** 2

        # Bias-corrected learning rate
        eta_norm = (
            self.eta_server
            * torch.sqrt(torch.tensor(1 - self.beta2_server ** t))
            / (1 - self.beta1_server ** t)
        )
        # x^{t+1} = x^t + η * m_t / (√v_t + τ)  (update from pre-round params, not FedAvg)
        new_params = OrderedDict(
            (k, prev[k] + eta_norm * self.m_t[k] / (torch.sqrt(self.v_t[k]) + self.tau_server))
            for k in prev
        )
        self._commit_global(new_params)


class FedAdam_Client(tFL_Client):
    pass
