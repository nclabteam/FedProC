import copy
import math
from argparse import Namespace
from collections import OrderedDict
from typing import Dict, List

import torch

from .base import SharedMethods
from .pFL import pFL, pFL_Client


class FedAMP(pFL):
    """
    FedAMP: Federated Learning with Attentive Message Passing.

    Server computes per-client attention-weighted mixtures of all uploaded
    models using pairwise L2 similarity: coef_ij ∝ exp(-||w_i-w_j||²/σ).
    Each client receives its personalized mixture and trains locally with a
    proximal regularization term anchored to that mixture.

    Reference: Huang et al., "Personalized Cross-Silo Federated Learning on
    Non-IID Data", AAAI 2021. arXiv 2007.03797.
    """

    optional = {
        "alphaK": 1.0,
        "sigma": 1.0,
        "lamda": 1.0,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--alphaK", type=float, default=None)
        parser.add_argument("--sigma", type=float, default=None)
        parser.add_argument("--lamda", type=float, default=None)

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        self.parallel = False
        # cid → OrderedDict[str, Tensor]: each client's latest uploaded model params
        self._uploaded: Dict[int, OrderedDict] = {}

    def _vec(self, params: OrderedDict) -> torch.Tensor:
        return torch.cat([v.float().flatten() for v in params.values()])

    def _attention(self, sq_dist: float) -> float:
        return math.exp(-sq_dist / self.sigma) / self.sigma

    def _compute_mixture(self, client_id: int) -> OrderedDict:
        """Compute attention-weighted mixture for client_id from all uploaded models.

        Falls back to public_model_params on round 0 (no uploads yet).
        """
        if not self._uploaded:
            return copy.deepcopy(self.public_model_params)

        wi_params = self._uploaded.get(client_id, self.public_model_params)
        wi = self._vec(wi_params)

        coefs: Dict[int, float] = {}
        for cid, params in self._uploaded.items():
            if cid == client_id:
                coefs[cid] = 0.0
            else:
                wj = self._vec(params)
                sq_dist = float(torch.dot(wi - wj, wi - wj))
                coefs[cid] = self.alphaK * self._attention(sq_dist)

        coef_self = 1.0 - sum(coefs.values())
        mixture = OrderedDict()
        for name in wi_params:
            acc = coef_self * wi_params[name].float()
            for cid, coef in coefs.items():
                if coef != 0.0:
                    acc = acc + coef * self._uploaded[cid][name].float()
            mixture[name] = acc.to(wi_params[name].dtype)
        return mixture

    def package(self, client_id: int) -> dict:
        pkg = super().package(client_id)
        # Replace global model with per-client attention mixture
        pkg["regular_model_params"] = self._compute_mixture(client_id)
        # Client must start from the mixture only — no personal overlay
        pkg["personal_model_params"] = {}
        return pkg

    def aggregate_client_updates(self, packages: "OrderedDict[int, dict]") -> None:
        for cid, pkg in packages.items():
            params = pkg["regular_model_params"]
            self._uploaded[cid] = copy.deepcopy(params)
            # Store trained model as personal params so pFL eval uses it
            self.clients_personal_model_params[cid] = dict(params)

        # Maintain a dummy global model (mean of all uploaded) for server bookkeeping
        all_params = list(self._uploaded.values())
        if all_params:
            new_global = OrderedDict()
            for name in all_params[0]:
                new_global[name] = torch.stack(
                    [p[name].float() for p in all_params]
                ).mean(dim=0).to(all_params[0][name].dtype)
            self._commit_global(new_global)


class FedAMP_Client(pFL_Client):
    """
    Client for FedAMP. Receives a personalized mixture u_i from the server and
    trains with proximal regularization: loss += (λ / 2αK) * ||w - u_i||².
    """

    def set_parameters(self, package: dict) -> None:
        super().set_parameters(package)
        # Anchor for proximal term: received mixture params in parameter order (CPU)
        self._u_params: List[torch.Tensor] = [
            v.clone().cpu() for v in package["regular_model_params"].values()
        ]

    def fit(self) -> None:
        SharedMethods._set_worker_seed(self._loader_seed("train"))
        loader = self.load_train_data()
        prox_coef = 0.5 * self.lamda / self.alphaK
        offload_after_epoch = self.efficiency == "low"

        for _ in range(self.epochs):
            # Mirror train_one_epoch: move to device + sync optimizer state each epoch
            self.model.to(self.device)
            SharedMethods._move_optimizer_state_to_param_devices(self.optimizer)
            self.model.train()
            for batch_x, batch_y, x_mark, y_mark in loader:
                self.optimizer.zero_grad(set_to_none=True)
                batch_x = batch_x.to(device=self.device, dtype=torch.float32, non_blocking=True)
                batch_y = batch_y.to(device=self.device, dtype=torch.float32, non_blocking=True)
                x_mark = x_mark.to(device=self.device, dtype=torch.float32, non_blocking=True)
                y_mark = y_mark.to(device=self.device, dtype=torch.float32, non_blocking=True)
                outputs = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
                loss = self.loss(outputs, batch_y)
                # Proximal term: (λ / 2αK) * ||w - u_i||²
                for p, u in zip(self.model.parameters(), self._u_params):
                    loss = loss + prox_coef * torch.norm(p - u.to(self.device), p=2) ** 2
                loss.backward()
                self.optimizer.step()
            if offload_after_epoch:
                self.model.to("cpu")
            self.scheduler.step()
        if self.efficiency == "med":
            self.model.to("cpu")
