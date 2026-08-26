import copy
import math
from argparse import Namespace
from collections import OrderedDict
from typing import Any, Dict, List

import torch

from .pFL import pFL, pFL_Client


class FedAMPShared:
    @staticmethod
    def validate_hyperparameters(alpha_k: float, sigma: float, lamda: float) -> None:
        if alpha_k <= 0 or sigma <= 0 or lamda < 0:
            raise ValueError("FedAMP requires alphaK > 0, sigma > 0, and lamda >= 0")


class FedAMP(FedAMPShared, pFL):
    """FedAMP: Federated Learning with Attentive Message Passing."""

    optional = {
        "alphaK": 1.0,
        "sigma": 1.0,
        "lamda": 1.0,
    }

    @classmethod
    def args_update(cls, parser: Any) -> None:
        parser.add_argument("--alphaK", type=float, default=None)
        parser.add_argument("--sigma", type=float, default=None)
        parser.add_argument("--lamda", type=float, default=None)

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        self.validate_hyperparameters(
            alpha_k=self.alphaK, sigma=self.sigma, lamda=self.lamda
        )
        # cid → OrderedDict[str, Tensor]: each client's latest uploaded model params
        self._uploaded: Dict[int, OrderedDict] = {}

    @staticmethod
    def _vec(params: OrderedDict) -> torch.Tensor:
        return torch.cat([v.float().flatten() for v in params.values()])

    @staticmethod
    def _attention(sq_dist: float, sigma: float) -> float:
        return math.exp(-sq_dist / sigma) / sigma

    def select_clients(self) -> None:
        self._select_all_clients()

    def _compute_mixture(self, client_id: int) -> OrderedDict:
        """Compute attention-weighted mixture for client_id from all uploaded models."""
        if not self._uploaded:
            return copy.deepcopy(self.public_model_params)

        wi_params = self._uploaded.get(client_id, self.public_model_params)
        wi = self._vec(params=wi_params)

        coefs: Dict[int, float] = {}
        for cid, params in self._uploaded.items():
            if cid == client_id:
                coefs[cid] = 0.0
            else:
                wj = self._vec(params=params)
                sq_dist = float(torch.dot(wi - wj, wi - wj))
                coefs[cid] = self.alphaK * self._attention(
                    sq_dist=sq_dist, sigma=self.sigma
                )

        coef_self = 1.0 - sum(coefs.values())
        if coef_self < -1e-7:
            raise ValueError(
                "FedAMP alphaK is too large for non-negative attention weights"
            )
        coef_self = max(0.0, coef_self)
        mixture = OrderedDict()
        for name in wi_params:
            acc = coef_self * wi_params[name].float()
            for cid, coef in coefs.items():
                if coef != 0.0:
                    acc = acc + coef * self._uploaded[cid][name].float()
            mixture[name] = acc.to(wi_params[name].dtype)
        return mixture

    def package(self, client_id: int) -> dict:
        pkg = super().package(client_id=client_id)
        # Replace global model with per-client attention mixture
        pkg["regular_model_params"] = self._compute_mixture(client_id=client_id)
        # Client must start from the mixture only — no personal overlay
        pkg["personal_model_params"] = {}
        return pkg

    def aggregate_client_updates(self, packages: "OrderedDict[int, dict]") -> None:
        for cid, pkg in packages.items():
            params = pkg["regular_model_params"]
            self._uploaded[cid] = copy.deepcopy(params)
            # Store trained model as personal params so pFL eval uses it
            self.clients_personal_model_params[cid] = dict(params)

        # FedAMP has no shared model; retain a mean only for framework bookkeeping.
        all_params = list(self._uploaded.values())
        if all_params:
            self._commit_global(new_params=self.mean_models(models=all_params))


class FedAMP_Client(FedAMPShared, pFL_Client):
    """Client for FedAMP."""

    def __init__(self, configs: Namespace, times: int, device: str) -> None:
        super().__init__(configs=configs, times=times, device=device)
        self.validate_hyperparameters(
            alpha_k=self.alphaK, sigma=self.sigma, lamda=self.lamda
        )

    def set_parameters(self, package: dict) -> None:
        super().set_parameters(package=package)
        # Anchor for proximal term: received mixture params in parameter order (CPU)
        self._u_params: List[torch.Tensor] = [
            v.clone().cpu() for v in package["regular_model_params"].values()
        ]

    def fit(self) -> None:
        self._set_worker_seed(seed=self._loader_seed(dataset_type="train"))
        loader = self.load_train_data()
        self.initialize_scheduler(steps_per_epoch=len(loader))
        prox_coef = 0.5 * self.lamda / self.alphaK
        offload_after_epoch = self.efficiency == "low"
        anchors = [value.to(self.device) for value in self._u_params]

        for _ in range(self.epochs):
            # Mirror train_one_epoch: move to device + sync optimizer state each epoch
            self.model.to(self.device)
            self._move_optimizer_state_to_param_devices(optimizer=self.optimizer)
            self.model.train()
            for batch_x, batch_y, x_mark, y_mark in loader:
                self.optimizer.zero_grad(set_to_none=True)
                batch_x = batch_x.to(
                    device=self.device, dtype=torch.float32, non_blocking=True
                )
                batch_y = batch_y.to(
                    device=self.device, dtype=torch.float32, non_blocking=True
                )
                x_mark = x_mark.to(
                    device=self.device, dtype=torch.float32, non_blocking=True
                )
                y_mark = y_mark.to(
                    device=self.device, dtype=torch.float32, non_blocking=True
                )
                outputs = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
                loss = self.loss(outputs, batch_y)
                # Proximal term: (λ / 2αK) * ||w - u_i||²
                for parameter, anchor in zip(self.model.parameters(), anchors):
                    loss = loss + prox_coef * torch.sum((parameter - anchor) ** 2)
                loss.backward()
                self.optimizer.step()
                self.step_scheduler_batch(
                    scheduler=self.scheduler,
                    batch_data=batch_x,
                )
            if offload_after_epoch:
                self.model.to("cpu")
            self.step_scheduler_epoch(scheduler=self.scheduler)
        if self.efficiency == "med":
            self.model.to("cpu")
