import copy
import math
import time
from argparse import Namespace
from typing import Any, Dict, List

import torch

from .pFL import pFL, pFL_Client


class FedAMP(pFL):
    """
    FedAMP: Federated Learning with Attentive Message Passing.

    Server computes per-client attention-weighted mixtures of all uploaded
    models using pairwise cosine/L2 similarity: coef_ij ∝ exp(-||w_i-w_j||²/σ).
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
        self._uploaded_models: List[torch.nn.Module] = []
        self._uploaded_ids: List[int] = []

    def _attention(self, sq_dist: float) -> float:
        return math.exp(-sq_dist / self.sigma) / self.sigma

    def variables_to_be_sent(self) -> Dict[str, Any]:
        if not self._uploaded_models:
            # First round: send global model to everyone
            return {"model": self.model}

        # Compute attention-weighted personalized model for each client
        personalized: List[torch.nn.Module] = []
        for i, ci in enumerate(self.clients):
            wi = torch.cat([p.data.view(-1) for p in ci.model.parameters()])
            coefs = []
            for j, cj_model in zip(self._uploaded_ids, self._uploaded_models):
                if ci.id == j:
                    coefs.append(0.0)
                else:
                    wj = torch.cat([p.data.view(-1) for p in cj_model.parameters()])
                    sq_dist = float(torch.dot(wi - wj, wi - wj))
                    coefs.append(self.alphaK * self._attention(sq_dist))

            coef_self = 1.0 - sum(coefs)
            mu = copy.deepcopy(ci.model)
            for p in mu.parameters():
                p.data.zero_()
            for coef, model_j in zip(coefs, self._uploaded_models):
                for p_mu, p_j in zip(mu.parameters(), model_j.parameters()):
                    p_mu.data.add_(p_j.data, alpha=coef)
            for p_mu, p_ci in zip(mu.parameters(), ci.model.parameters()):
                p_mu.data.add_(p_ci.data, alpha=coef_self)
            personalized.append(mu)

        return {"model": personalized}

    def receive_from_clients(self) -> None:
        self.client_data = []
        self._uploaded_models = []
        self._uploaded_ids = []
        for client in self.selected_clients:
            try:
                data = client.send_to_server()
                self.client_data.append(data)
                self._uploaded_models.append(copy.deepcopy(client.model))
                self._uploaded_ids.append(client.id)
            except Exception as e:
                self.logger.error(f"Failed to receive from client {client.id}: {e}")

    def aggregate_models(self) -> None:
        # FedAMP has no global model aggregation — personalization is server-side
        pass


class FedAMP_Client(pFL_Client):
    """
    Client for FedAMP. Receives a personalized mixture model u_i from server
    and trains with proximal regularization: loss += (λ / 2αK) * ||w - u_i||².
    """

    def receive_from_server(self, data: dict) -> None:
        self._u_model = copy.deepcopy(data["model"]).to("cpu")
        self.update_model_params(old=self.model, new=data["model"])

    def train(self):
        train_loader = self.load_train_data()
        start_time = time.time()

        prox_coef = 0.5 * self.lamda / self.alphaK
        u_params = [p.data.clone() for p in self._u_model.parameters()]

        self.model.to(self.device)
        self.model.train()
        offload_after = self.efficiency == "low"
        for _ in range(self.epochs):
            for batch_x, batch_y, x_mark, y_mark in train_loader:
                self.optimizer.zero_grad()
                batch_x = batch_x.to(device=self.device, dtype=torch.float32)
                batch_y = batch_y.to(device=self.device, dtype=torch.float32)
                x_mark = x_mark.to(device=self.device, dtype=torch.float32)
                y_mark = y_mark.to(device=self.device, dtype=torch.float32)
                outputs = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
                loss = self.loss(outputs, batch_y)
                # Proximal term: (λ/2αK) * ||w - u_i||²
                for p, u in zip(self.model.parameters(), u_params):
                    loss = loss + prox_coef * torch.norm(p - u.to(self.device), p=2) ** 2
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            if offload_after:
                self.model.to("cpu")
        if self.efficiency == "med":
            self.model.to("cpu")
        self.metrics["train_time"].append(time.time() - start_time)
