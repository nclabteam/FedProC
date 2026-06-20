from collections import OrderedDict

import torch
import torch.nn as nn

from .dFL import dFL, dFL_Client
from .tFL import tFL as Server


class FedAWA(Server):

    optional = {
        "server_epochs": 1,
        "reg_distance": "cos",
        "server_lr": 0.01,
        "server_optimizer": "Adam",
        "fedawa_gamma": 1.0,
    }

    awa_weights = None
    _awa_optimizer = None

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--server_epochs", type=int, default=None)
        parser.add_argument(
            "--reg_distance", type=str, default=None, choices=["cos", "euc"]
        )
        parser.add_argument("--server_lr", type=float, default=None)
        parser.add_argument(
            "--server_optimizer", type=str, default=None, choices=["SGD", "Adam"]
        )
        parser.add_argument("--fedawa_gamma", type=float, default=None)

    @staticmethod
    def _flatten_params(model):
        """Helper function to flatten model parameters into a single vector."""
        return torch.cat([p.data.view(-1) for p in model.parameters()])

    @staticmethod
    def _cost_matrix(x, y, dis="cos", p=2):
        """
        Calculates the cost matrix between representations x and y.
        Adapted from the original FedAWA code.

        Args:
            x: Tensor (e.g., global model flat params), shape [..., features]
            y: Tensor (e.g., client model flat params), shape [..., features]
            dis: Distance metric ('cos' or 'euc').
            p: Power for Euclidean distance.

        Returns:
            Tensor: Cost matrix.
        """
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        if torch.is_complex(x_col):
            x_col = x_col.real
        if torch.is_complex(y_lin):
            y_lin = y_lin.real
        if dis == "cos":
            d_cosine = nn.CosineSimilarity(dim=-1, eps=1e-8)
            C = 1 - d_cosine(x_col, y_lin)
        elif dis == "euc":
            C = torch.mean((torch.abs(x_col - y_lin)) ** p, -1)
        else:
            raise ValueError(f"Unsupported distance type: {dis}")
        return C

    def aggregate_client_updates(self, packages) -> None:
        cids = list(packages.keys())
        num_clients = len(cids)
        scores = [packages[cid]["score"] for cid in cids]

        client_params = [packages[cid]["regular_model_params"] for cid in cids]
        client_flats = torch.stack([
            torch.cat([p.view(-1).float() for p in params.values()])
            for params in client_params
        ]).to(self.device)

        global_flat = torch.cat(
            [p.view(-1).float() for p in self.public_model_params.values()]
        ).to(self.device)

        if self.awa_weights is None or self.awa_weights.shape[0] != num_clients:
            ts = torch.tensor(scores, dtype=torch.float32, device=self.device)
            self.awa_weights = torch.log(ts + 1e-9).clone().detach().requires_grad_(True)
            obj = self._get_objective_function("optimizers", self.server_optimizer)
            self._awa_optimizer = obj(
                params=[self.awa_weights],
                configs=type(
                    "Config",
                    (),
                    {
                        "learning_rate": self.server_lr,
                        "momentum": 0.9,
                        "beta1": 0.9,
                        "beta2": 0.999,
                        "epsilon": 1e-8,
                        "weight_decay": 0,
                        "amsgrad": False,
                    },
                )(),
            )
        else:
            self.awa_weights = self.awa_weights.clone().detach().requires_grad_(True)

        self.awa_weights = self.awa_weights.to(self.device).requires_grad_(True)

        for _ in range(self.server_epochs):
            self._awa_optimizer.zero_grad()

            probability_train = torch.nn.functional.softmax(self.awa_weights, dim=0)

            C = self._cost_matrix(
                x=global_flat.unsqueeze(0),
                y=client_flats,
                dis=self.reg_distance,
            )
            reg_loss = torch.sum(probability_train * C.squeeze(0))

            client_updates = client_flats - global_flat
            weighted_avg_update = torch.sum(
                client_updates * probability_train.unsqueeze(1), dim=0, keepdim=True
            )
            l2_distance = torch.norm(client_updates - weighted_avg_update, p=2, dim=1)
            sim_loss = torch.sum(probability_train * l2_distance)

            total_loss = sim_loss + reg_loss
            total_loss.backward()
            self._awa_optimizer.step()

        final_logits = self.awa_weights.detach().clone()
        self.awa_weights = final_logits

        final_probabilities = torch.nn.functional.softmax(final_logits, dim=0)
        scaled_weights = final_probabilities * self.fedawa_gamma
        sum_scaled = scaled_weights.sum()
        if sum_scaled > 1e-8:
            weights = scaled_weights / sum_scaled
        else:
            weights = torch.ones_like(final_probabilities) / num_clients

        new_params = OrderedDict()
        for name in self.public_model_params:
            stacked = torch.stack(
                [client_params[i][name].float() for i in range(num_clients)], dim=-1
            )
            new_params[name] = torch.sum(
                stacked * weights.to(stacked.dtype), dim=-1
            ).to(self.public_model_params[name].dtype)
        self._commit_global(new_params)


class DFedAWA(FedAWA, dFL):
    """Decentralized FedAWA: learn adaptive aggregation weights per receiver."""

    @classmethod
    def args_update(cls, parser):
        dFL.args_update(parser)
        FedAWA.args_update(parser)

    def calculate_aggregation_weights(self):
        dFL.calculate_aggregation_weights(
            self
        )  # delegate to dFL's client-orchestrating version


class DFedAWA_Client(dFL_Client):
    _flatten_params = staticmethod(FedAWA._flatten_params)
    _cost_matrix = staticmethod(FedAWA._cost_matrix)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.awa_weights = None
        self.awa_optimizer = None

    def calculate_aggregation_weights(self):
        self.client_data = [
            {"model": model, "score": score}
            for model, score in zip(self.models, self.scores)
        ]
        model_optimizer = self.optimizer
        self.optimizer = self.awa_optimizer
        FedAWA.calculate_aggregation_weights(self)
        self.awa_optimizer = self.optimizer
        self.optimizer = model_optimizer
