from collections import OrderedDict

import torch

from .tFL import tFL


class FedAdam(tFL):

    optional = {
        "beta1_server": 1e-1,
        "beta2_server": 0.999,
        "eta_server": 0.001,
        "tau_server": 1e-9,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--beta1_server", type=float, default=None)
        parser.add_argument("--beta2_server", type=float, default=None)
        parser.add_argument("--eta_server", type=float, default=None)
        parser.add_argument("--tau_server", type=float, default=None)

    def aggregate_client_updates(self, packages):
        prev = OrderedDict((k, v.clone()) for k, v in self.public_model_params.items())
        super().aggregate_client_updates(packages)  # FedAvg step

        delta = {k: prev[k] - self.public_model_params[k] for k in prev}
        if not hasattr(self, "m_t"):
            self.m_t = {k: torch.zeros_like(v) for k, v in prev.items()}
            self.v_t = {k: torch.zeros_like(v) for k, v in prev.items()}

        for k in prev:
            self.m_t[k] = (
                self.beta1_server * self.m_t[k]
                + (1 - self.beta1_server) * delta[k]
            )
            self.v_t[k] = (
                self.beta2_server * self.v_t[k]
                + (1 - self.beta2_server) * delta[k] ** 2
            )

        eta_norm = (
            self.eta_server
            * torch.sqrt(
                torch.tensor(1 - self.beta2_server ** (self.current_iter + 1.0))
            )
            / (1 - self.beta1_server ** (self.current_iter + 1.0))
        )
        new_params = OrderedDict(
            (
                k,
                self.public_model_params[k]
                + eta_norm * self.m_t[k] / (torch.sqrt(self.v_t[k]) + self.tau_server),
            )
            for k in prev
        )
        self._commit_global(new_params)
