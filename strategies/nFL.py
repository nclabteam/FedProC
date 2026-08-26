import copy
from typing import Any

import torch

from .pFL import pFL, pFL_Client


class nFLShared:
    """Operations shared by non-federated representation-learning clients."""

    @staticmethod
    def fit_ridge_head(model: Any, dataloader: Any, device: Any, alpha: float) -> None:
        """Fit the frozen encoder's linear forecasting head by ridge regression."""
        if alpha < 0:
            raise ValueError("ridge_alpha must be non-negative")

        model.to(device)
        model.eval()
        feature_dim = model.head.in_features
        target_dim = model.head.out_features
        gram = torch.zeros(feature_dim + 1, feature_dim + 1, dtype=torch.float64)
        cross = torch.zeros(feature_dim + 1, target_dim, dtype=torch.float64)
        sample_count = 0

        with torch.no_grad():
            for batch_x, batch_y, *_ in dataloader:
                features = (
                    model.representation(batch_x.to(device=device, dtype=torch.float32))
                    .detach()
                    .cpu()
                    .to(torch.float64)
                )
                targets = batch_y.reshape(batch_y.shape[0], -1).to(torch.float64)
                design = torch.cat(
                    [features, torch.ones(features.shape[0], 1, dtype=features.dtype)],
                    dim=1,
                )
                gram.add_(design.T @ design)
                cross.add_(design.T @ targets)
                sample_count += features.shape[0]

        if sample_count == 0:
            raise ValueError("cannot fit a ridge head on an empty dataset")
        penalty = torch.eye(feature_dim + 1, dtype=gram.dtype) * alpha
        penalty[-1, -1] = 0
        weights = torch.linalg.solve(gram + penalty, cross).to(
            dtype=model.head.weight.dtype
        )
        with torch.no_grad():
            model.head.weight.copy_(weights[:-1].T.to(model.head.weight.device))
            model.head.bias.copy_(weights[-1].to(model.head.bias.device))


class nFL(pFL):
    """No-federation base: each client persists and evaluates its own model."""

    def __init__(self, configs: Any, times: Any) -> None:
        super().__init__(configs=configs, times=times)
        self.clients_auxiliary_state = {
            client_id: {} for client_id in range(self.num_clients)
        }

    def package(self, client_id: int) -> Any:
        package = super().package(client_id=client_id)
        package["auxiliary_state"] = copy.deepcopy(
            self.clients_auxiliary_state[client_id]
        )
        return package

    def _compute_send_mb(self, packages: Any) -> tuple:
        return {}, 0.0

    def aggregate_client_updates(self, packages: Any) -> None:
        for client_id, package in packages.items():
            self.clients_personal_model_params[client_id].update(
                package["regular_model_params"]
            )
            self.clients_auxiliary_state[client_id] = copy.deepcopy(
                package.get("auxiliary_state", {})
            )

    def _pre_eval_hook(self, dataset_type: str) -> None:
        """Skip the personalized pre-evaluation hook."""

    def evaluate_generalization(self, dataset_type: str) -> None:
        pFL._pre_eval_hook(self=self, dataset_type=dataset_type)


class nFL_Client(nFLShared, pFL_Client):
    """Shared independent-client implementation for non-federated methods."""

    def set_parameters(self, package: Any) -> None:
        self.auxiliary_state = copy.deepcopy(package.get("auxiliary_state", {}))
        super().set_parameters(package=package)

    def package(self) -> Any:
        package = super().package()
        package["auxiliary_state"] = copy.deepcopy(self.auxiliary_state)
        return package
