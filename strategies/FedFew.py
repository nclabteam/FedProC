# -*- coding: utf-8 -*-
"""FedFew: Few-for-Many Personalized Federated Learning."""

import copy
from collections import OrderedDict
from typing import Any, Dict, List

import torch

from .pFL import pFL, pFL_Client


class FedFewShared:
    """Stable STCH-Set operations shared by FedFew server and client."""

    @staticmethod
    def stch_weights(losses: torch.Tensor, scores: torch.Tensor, mu: float) -> Any:
        if mu <= 0:
            raise ValueError("FedFew mu must be positive")
        if losses.ndim != 2 or scores.ndim != 1 or len(losses) != len(scores):
            raise ValueError("FedFew expects MxK losses and M sample counts")
        if scores.sum() <= 0:
            raise ValueError("FedFew sample counts must sum to a positive value")
        weighted_losses = losses * (scores / scores.sum()).unsqueeze(1)
        logits = -weighted_losses / mu
        log_s = torch.logsumexp(logits, dim=1)
        return torch.softmax(-log_s, dim=0), torch.softmax(logits, dim=1)


class FedFew(FedFewShared, pFL):
    """FedFew server."""

    optional = {"num_models": 3, "mu": 0.01}

    @classmethod
    def args_update(cls, parser: Any) -> Any:
        parser.add_argument("--num_models", type=int, default=None)
        parser.add_argument("--mu", type=float, default=None)
        return parser

    def __init__(self, configs: Any, times: Any) -> None:
        super().__init__(configs=configs, times=times)
        self.server_models: List[OrderedDict] = [
            OrderedDict({k: v.clone() for k, v in self.public_model_params.items()})
            for _ in range(self.num_models)
        ]

    def package(self, client_id: int) -> dict:
        pkg = super().package(client_id=client_id)
        pkg["fedfew_server_models"] = self.server_models
        pkg["__wire__"] = ("fedfew_server_models",)
        return pkg

    def select_clients(self) -> None:
        self._select_all_clients()

    def aggregate_client_updates(self, packages: Any) -> None:
        cids = list(packages.keys())
        if not cids:
            raise ValueError("FedFew requires at least one client")
        losses = torch.tensor(
            [packages[cid]["fedfew_losses"] for cid in cids], dtype=torch.float64
        )
        scores = torch.tensor(
            [packages[cid]["score"] for cid in cids], dtype=torch.float64
        )
        alpha, model_weights = self.stch_weights(
            losses=losses, scores=scores, mu=self.mu
        )
        client_weights = alpha.unsqueeze(1) * model_weights

        # θ_k^t = θ_k^{t-1} - lr · Σ_i α_i w_{ik} g_{ik}
        for k in range(self.num_models):
            for name in self.server_models[k]:
                gradients = torch.stack(
                    [packages[cid]["fedfew_gradients"][k][name] for cid in cids]
                ).float()
                gradient = torch.tensordot(
                    client_weights[:, k].to(gradients), gradients, dims=([0], [0])
                )
                self.server_models[k][name] = (
                    self.server_models[k][name] - self.learning_rate * gradient
                )

        # Selection is evaluation-only during training; Algorithm 1 deploys it at T.
        for index, cid in enumerate(cids):
            best_k = int(losses[index].argmin())
            self.clients_personal_model_params[cid].update(
                {k: v.clone() for k, v in self.server_models[best_k].items()}
            )

        # Integration-only summary used by FedProC's generalization metric.
        self._commit_global(new_params=self.mean_models(models=self.server_models))


class FedFew_Client(FedFewShared, pFL_Client):
    """FedFew client — trains all K server models locally."""

    _server_models: List[OrderedDict] = None
    _fedfew_losses: List[float] = None
    _fedfew_gradients: List[Dict[str, torch.Tensor]] = None

    def set_parameters(self, package: dict) -> None:
        super().set_parameters(package=package)
        self._server_models = package["fedfew_server_models"]

    def fit(self) -> None:
        self._fedfew_losses = []
        self._fedfew_gradients = []

        for model_params in self._server_models:
            model_k = copy.deepcopy(self.model).to(self.device)
            model_k.load_state_dict(model_params, strict=False)
            optimizer_k = torch.optim.SGD(model_k.parameters(), lr=self.learning_rate)

            loader = self.load_train_data()
            model_k.train()
            for _ in range(self.epochs):
                for batch in loader:
                    optimizer_k.zero_grad(set_to_none=True)
                    self._batch_loss(model=model_k, batch=batch).backward()
                    optimizer_k.step()

            # Paper Algorithm 1, pseudocode step 7: g_ik = grad L_i(theta_k).
            model_k.zero_grad(set_to_none=True)
            num_samples = 0
            total_loss = 0.0
            for batch in loader:
                batch_size = len(batch[0])
                loss = self._batch_loss(model=model_k, batch=batch)
                (loss * batch_size).backward()
                total_loss += loss.item() * batch_size
                num_samples += batch_size

            scale = 1.0 / max(num_samples, 1)
            self._fedfew_losses.append(total_loss * scale)
            self._fedfew_gradients.append(
                {
                    n: (
                        p.grad.detach().cpu() * scale
                        if p.grad is not None
                        else torch.zeros_like(p.data).cpu()
                    )
                    for n, p in model_k.named_parameters()
                }
            )

    def _batch_loss(self, model: Any, batch: Any) -> Any:
        batch_x, batch_y, x_mark, y_mark = [
            value.to(self.device, dtype=torch.float32) for value in batch
        ]
        return self.loss(
            model(batch_x, x_mark=x_mark, y_mark=y_mark),
            batch_y,
        )

    def package(self) -> Dict[str, Any]:
        result = super().package()
        result["fedfew_losses"] = self._fedfew_losses
        result["fedfew_gradients"] = self._fedfew_gradients
        result["__wire__"] = ("fedfew_gradients", "fedfew_losses", "score")
        return result
