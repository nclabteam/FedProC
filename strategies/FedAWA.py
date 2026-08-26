from argparse import Namespace
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .dFL import dFL, dFL_Client
from .tFL import tFL


class FedAWAShared:
    """FedAWA operations shared by central and decentralized aggregation."""

    @staticmethod
    def flatten_params(params: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([value.reshape(-1).float() for value in params.values()])

    @staticmethod
    def cost_matrix(
        x: torch.Tensor,
        y: torch.Tensor,
        distance: str = "cos",
        power: int = 2,
    ) -> torch.Tensor:
        x_col, y_line = x.unsqueeze(-2), y.unsqueeze(-3)
        if torch.is_complex(x_col):
            x_col = x_col.real
        if torch.is_complex(y_line):
            y_line = y_line.real
        if distance == "cos":
            return 1 - F.cosine_similarity(x_col, y_line, dim=-1, eps=1e-8)
        if distance == "euc":
            return torch.mean(torch.abs(x_col - y_line) ** power, dim=-1)
        raise ValueError(f"Unsupported distance type: {distance}")

    @staticmethod
    def optimize_weights(
        models: Sequence[Mapping[str, torch.Tensor]],
        reference: Mapping[str, torch.Tensor],
        initial_logits: torch.Tensor,
        epochs: int,
        learning_rate: float,
        optimizer_name: str,
        distance: str,
        device: str | torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Optimize aggregation weights."""
        if not models:
            raise ValueError("at least one model is required")
        if initial_logits.numel() != len(models):
            raise ValueError("one initial logit per model is required")

        local = torch.stack(
            [FedAWAShared.flatten_params(params=model) for model in models]
        ).to(device=device)
        old = FedAWAShared.flatten_params(params=reference).to(device=device)
        logits = initial_logits.detach().clone().to(device=device).requires_grad_(True)
        if optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam([logits], lr=learning_rate, betas=(0.5, 0.999))
        elif optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(
                [logits], lr=learning_rate, momentum=0.9, weight_decay=5e-4
            )
        else:
            raise ValueError(f"Unsupported server optimizer: {optimizer_name}")

        updates = local - old
        for _ in range(epochs):
            optimizer.zero_grad()
            weights = F.softmax(logits, dim=0)
            merged_update = torch.matmul(weights, updates)
            # Paper Eq. 3: sum_k lambda_k ||tau_k - tau_g||_2.
            similarity = torch.dot(
                weights,
                torch.linalg.vector_norm(updates - merged_update, dim=1),
            )
            merged_model = torch.matmul(weights, local)
            # Paper Eq. 3: d(sum_k lambda_k theta_k, theta_g).
            regularizer = FedAWAShared.cost_matrix(
                x=merged_model.unsqueeze(0),
                y=old.unsqueeze(0),
                distance=distance,
            ).squeeze()
            (similarity + regularizer).backward()
            optimizer.step()

        logits = logits.detach().cpu()
        return F.softmax(logits, dim=0), logits


class FedAWA(FedAWAShared, tFL):
    """Adaptive server aggregation using client update vectors."""

    optional = {
        "server_epochs": 1,
        "reg_distance": "cos",
        "server_lr": 0.001,
        "server_optimizer": "Adam",
    }

    @classmethod
    def args_update(cls, parser: Any) -> None:
        parser.add_argument("--server_epochs", type=int, default=None)
        parser.add_argument(
            "--reg_distance", type=str, default=None, choices=["cos", "euc"]
        )
        parser.add_argument("--server_lr", type=float, default=None)
        parser.add_argument(
            "--server_optimizer", type=str, default=None, choices=["SGD", "Adam"]
        )

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        self.awa_weights: dict[Any, torch.Tensor] = {}

    def aggregate_client_updates(self, packages: Mapping[int, dict[str, Any]]) -> None:
        client_ids = list(packages)
        models = [
            packages[client_id]["regular_model_params"] for client_id in client_ids
        ]
        initial_logits = torch.stack(
            [
                self.awa_weights.get(
                    client_id,
                    torch.tensor(max(float(packages[client_id]["score"]), 1e-12)).log(),
                )
                for client_id in client_ids
            ]
        )
        weights, logits = self.optimize_weights(
            models=models,
            reference=self.public_model_params,
            initial_logits=initial_logits,
            epochs=self.server_epochs,
            learning_rate=self.server_lr,
            optimizer_name=self.server_optimizer,
            distance=self.reg_distance,
            device=self.device,
        )
        self.awa_weights.update(
            {client_id: logit.clone() for client_id, logit in zip(client_ids, logits)}
        )
        self._commit_global(new_params=self.mean_models(models=models, weights=weights))


class DFedAWA(FedAWA, dFL):
    """Apply FedAWA independently over each node's neighborhood."""

    @classmethod
    def args_update(cls, parser: Any) -> None:
        dFL.args_update(parser=parser)
        FedAWA.args_update(parser=parser)

    def train_one_round(self) -> dict[str, Any]:
        self._round_reference_models = {
            client_id: dict(
                self.clients_personal_model_params[client_id]
                or self.public_model_params
            )
            for client_id in self.selected_clients
        }
        return super().train_one_round()

    def aggregate_client_updates(self, packages: Mapping[int, dict[str, Any]]) -> None:
        trained = {
            client_id: package["regular_model_params"]
            for client_id, package in packages.items()
        }
        references = getattr(self, "_round_reference_models", {})
        aggregated = {}
        for client_id in packages:
            peers = [
                client_id,
                *(peer for peer in self.topology[client_id] if peer in packages),
            ]
            models = [trained[peer] for peer in peers]
            initial_logits = torch.stack(
                [
                    self.awa_weights.get(
                        (client_id, peer),
                        torch.tensor(max(float(packages[peer]["score"]), 1e-12)).log(),
                    )
                    for peer in peers
                ]
            )
            reference = references.get(client_id) or (
                self.clients_personal_model_params[client_id]
                or self.public_model_params
            )
            weights, logits = self.optimize_weights(
                models=models,
                reference=reference,
                initial_logits=initial_logits,
                epochs=self.server_epochs,
                learning_rate=self.server_lr,
                optimizer_name=self.server_optimizer,
                distance=self.reg_distance,
                device=self.device,
            )
            self.awa_weights.update(
                {(client_id, peer): logit.clone() for peer, logit in zip(peers, logits)}
            )
            aggregated[client_id] = self.mean_models(models=models, weights=weights)

        for client_id, model in aggregated.items():
            self.clients_personal_model_params[client_id].update(model)


class DFedAWA_Client(dFL_Client):
    """Stateless local trainer for decentralized FedAWA."""
