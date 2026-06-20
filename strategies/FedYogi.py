from collections import OrderedDict

import torch

from .tFL import tFL, tFL_Client


class FedYogi(tFL):
    """FedYogi — server-side Yogi optimizer for federated learning.

    Like FedAdam but uses the Yogi second-moment update, which reduces
    v_t only when the current squared gradient exceeds v_t (prevents
    over-shooting in well-explored directions).

    Reference: Reddi et al., "Adaptive Federated Optimization", ICLR 2021.
    arXiv:2003.00295.
    """

    optional = {
        "beta1_server": 0.9,
        "beta2_server": 0.999,
        "eta_server": 1e-2,
        "tau_server": 1e-3,
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
            # Yogi: v_t -= (1-β2) * sign(v_t - Δ^2) * Δ^2
            delta_sq = delta[k] ** 2
            self.v_t[k] = self.v_t[k] - (1 - self.beta2_server) * torch.sign(
                self.v_t[k] - delta_sq
            ) * delta_sq

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


class FedYogi_Client(tFL_Client):
    pass
