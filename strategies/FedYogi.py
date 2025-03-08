import copy

import torch

from .base import Server

optional = {
    "beta1_server": 0.9,
    "beta2_server": 0.999,
    "eta_server": 1e-2,
    "tau_server": 1e-3,
}


def args_update(parser):
    parser.add_argument(
        "--beta1_server",
        type=float,
        default=None,
        help="Beta1 parameter for Adam optimizer on the server",
    )
    parser.add_argument(
        "--beta2_server",
        type=float,
        default=None,
        help="Beta2 parameter for Adam optimizer on the server",
    )
    parser.add_argument(
        "--eta_server",
        type=float,
        default=None,
        help="Learning rate for Adam optimizer on the server",
    )
    parser.add_argument(
        "--tau_server",
        type=float,
        default=None,
        help="Controls the algorithm's degree of adaptability",
    )


class FedYogi(Server):
    """
    Paper: https://arxiv.org/abs/2003.00295
    Source: https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedyogi.py
    """

    def aggregate_models(self):
        prev_model = copy.deepcopy(self.model)

        # Call the base class method (FedAvg)
        super().aggregate_models()

        delta_t = copy.deepcopy(self.model)
        for curr_param, prev_param, diff in zip(
            self.model.parameters(), prev_model.parameters(), delta_t.parameters()
        ):
            diff.data = prev_param.data - curr_param.data

        if not hasattr(self, "m_t"):
            self.m_t = self.reset_model(self.model)

        if not hasattr(self, "v_t"):
            self.v_t = self.reset_model(self.model)

        for m_t, v_t, grad in zip(
            self.m_t.parameters(), self.v_t.parameters(), delta_t.parameters()
        ):
            m_t.data = (
                self.beta1_server * m_t.data + (1 - self.beta1_server) * grad.data
            )
            v_t.data = v_t.data - (1 - self.beta2_server) * (
                grad.data**2 - v_t.data
            ) * torch.sign(v_t.data - grad.data**2)

        eta_norm = (
            self.eta_server
            * torch.sqrt(
                torch.tensor(1 - self.beta2_server ** (self.current_iter + 1.0))
            )
            / (1 - self.beta1_server ** (self.current_iter + 1.0))
        )

        for param, m_t, v_t in zip(
            self.model.parameters(), self.m_t.parameters(), self.v_t.parameters()
        ):
            param.data = param.data + eta_norm * m_t.data / (
                torch.sqrt(v_t.data) + self.tau_server
            )
