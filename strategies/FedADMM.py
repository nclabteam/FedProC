import copy
import time
from argparse import Namespace
from typing import Any, Dict

import torch

from .tFL import tFL, tFL_Client


class FedADMM(tFL):
    """
    FedADMM: Federated Learning via Alternating Direction Method of Multipliers.

    Each client maintains a Lagrangian multiplier α per parameter. During local
    training the gradient is augmented with the ADMM correction term
    α + ρ·(w - θ), where θ is the server's consensus variable. After training,
    α is updated by α += ρ·(w_new - θ). The client sends
    local_sum = (w_new - w_prev) + (1/ρ)·(α_new - α_prev) to the server.
    The server updates θ += η · weighted_avg(local_sum) and broadcasts θ.

    Reference: fl-bench FedADMM implementation.
    """

    optional = {
        "rho": 0.01,
        "eta": 1.0,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--rho", type=float, default=None)
        parser.add_argument("--eta", type=float, default=None)

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        self.parallel = False
        self.theta: Dict[str, torch.Tensor] = {
            name: p.data.clone().cpu()
            for name, p in self.model.named_parameters()
        }

    def aggregate_models(self) -> None:
        total_score = sum(cd["score"] for cd in self.client_data)

        # Update theta: θ += η · weighted_avg(local_sum)
        with torch.no_grad():
            for name in self.theta:
                update = sum(
                    cd["local_sum"][name].cpu() * (cd["score"] / total_score)
                    for cd in self.client_data
                    if name in cd["local_sum"]
                )
                self.theta[name] = self.theta[name] + self.eta * update

            # Sync model weights from theta
            for name, param in self.model.named_parameters():
                param.data.copy_(self.theta[name])


class FedADMM_Client(tFL_Client):
    """
    Client for FedADMM. Maintains per-parameter Lagrangian multipliers α.
    Applies ADMM gradient correction during training and returns local_sum.
    """

    def __init__(self, configs: Namespace, id: int, times: int) -> None:
        super().__init__(configs=configs, id=id, times=times)
        self.alpha: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(p, device="cpu")
            for name, p in self.model.named_parameters()
        }
        self._theta: Dict[str, torch.Tensor] = {}
        self._model_prev: Dict[str, torch.Tensor] = {}
        self._alpha_prev: Dict[str, torch.Tensor] = {}

    def receive_from_server(self, data: dict) -> None:
        self.update_model_params(old=self.model, new=data["model"])
        self._theta = {
            name: p.data.clone().cpu()
            for name, p in data["model"].named_parameters()
        }
        self._model_prev = {
            name: p.data.clone().cpu()
            for name, p in self.model.named_parameters()
        }
        self._alpha_prev = {name: a.clone() for name, a in self.alpha.items()}

    def train(self):
        train_loader = self.load_train_data()
        start_time = time.time()

        self.model.to(self.device)
        self.model.train()

        for _ in range(self.epochs):
            for batch_x, batch_y, x_mark, y_mark in train_loader:
                self.optimizer.zero_grad()
                batch_x = batch_x.to(device=self.device, dtype=torch.float32)
                batch_y = batch_y.to(device=self.device, dtype=torch.float32)
                x_mark = x_mark.to(device=self.device, dtype=torch.float32)
                y_mark = y_mark.to(device=self.device, dtype=torch.float32)
                outputs = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
                loss = self.loss(outputs, batch_y)
                loss.backward()

                # ADMM gradient correction: grad += α + ρ·(w - θ)
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.grad is None:
                            continue
                        alpha_i = self.alpha[name].to(self.device)
                        theta_i = self._theta[name].to(self.device)
                        param.grad.add_(alpha_i + self.rho * (param.data - theta_i))

                self.optimizer.step()
            self.scheduler.step()

        # Update α: α += ρ·(w_new - θ)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                theta_i = self._theta[name].to(self.device)
                self.alpha[name] = (
                    self.alpha[name].to(self.device) + self.rho * (param.data - theta_i)
                ).cpu()

        self.model.to("cpu")
        self.metrics["train_time"].append(time.time() - start_time)

    def variables_to_be_sent(self) -> Dict[str, Any]:
        local_sum: Dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            w_new = param.data.cpu()
            w_prev = self._model_prev[name]
            alpha_new = self.alpha[name]
            alpha_prev = self._alpha_prev[name]
            local_sum[name] = (w_new - w_prev) + (1.0 / self.rho) * (alpha_new - alpha_prev)
        return {"local_sum": local_sum, "score": self.train_samples}
