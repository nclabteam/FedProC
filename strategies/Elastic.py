import copy

import torch

from .base import Client, Server

optional = {
    "tau": 0.5,
    "sample_ratio": 0.3,
    "mu": 0.95,
}


def args_update(parser):
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--sample_ratio", type=float, default=None)
    parser.add_argument("--mu", type=float, default=None)


class Elastic(Server):
    """
    Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Elastic_Aggregation_for_Federated_Optimization_CVPR_2023_paper.html
    Source: https://github.com/KarhouTam/FL-bench/blob/master/src/server/elastic.py
    """

    def calculate_aggregation_weights(self):
        super().calculate_aggregation_weights()
        sensitivities = torch.stack(
            [client["sensitivity"] for client in self.client_data], dim=-1
        )
        aggregated_sensitivity = torch.sum(sensitivities * self.weights, dim=-1)
        max_sensitivity = sensitivities.max(dim=-1)[0]
        self.zeta = 1 + self.tau - aggregated_sensitivity / max_sensitivity

    def aggregate_models(self):
        self.model = self.reset_model(self.model)
        for client, weight, coef in zip(self.client_data, self.weights, self.zeta):
            for global_param, local_param in zip(
                self.model.parameters(), client["model"].parameters()
            ):
                aggregated = local_param.data * weight
                global_param.data -= coef * aggregated

    def pre_train_clients(self):
        for client in self.selected_clients:
            client.calculate_sensitivity()


class Elastic_Client(Client):
    """
    Source: https://github.com/KarhouTam/FL-bench/blob/master/src/client/elastic.py
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_sensitivity = torch.zeros(
            len(list(self.model.parameters())), device=self.device
        )

    def variables_to_be_sent(self):
        to_be_sent = super().variables_to_be_sent()
        diff_dict = {
            key: param_old - param_new
            for (key, param_old), param_new in zip(
                self.snapshot.named_parameters(), self.model.parameters()
            )
        }
        diff_model = copy.deepcopy(self.model)
        diff_model.load_state_dict(diff_dict)
        to_be_sent["model"] = diff_model
        return {**to_be_sent, "sensitivity": self.sensitivity}

    def calculate_sensitivity(self):
        sensitivity = self.init_sensitivity
        self.model.eval()
        self.snapshot = copy.deepcopy(self.model)
        for x, y in self.load_train_data(sample_ratio=self.sample_ratio, shuffle=False):
            x = x.to(self.device)
            y = y.to(self.device)
            logits = self.model(x)
            loss = self.loss(logits, y)
            loss.backward()
            grad_norms = []
            for param in self.model.parameters():
                if param.requires_grad:
                    grad_norms.append(torch.norm(param.grad.data) ** 2)
                else:
                    grad_norms.append(None)
            for i in range(len(grad_norms)):
                if grad_norms[i]:
                    sensitivity[i] = (
                        self.mu * sensitivity[i] + (1 - self.mu) * grad_norms[i].abs()
                    )
                else:
                    sensitivity[i] = 1.0
        self.sensitivity = sensitivity
