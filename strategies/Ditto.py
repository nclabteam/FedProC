import copy
from typing import Any, Dict

import numpy as np
import torch

from .pFL import pFL, pFL_Client


class Ditto(pFL):
    """
    Ditto: Fair and Robust Personalized Federated Learning.

    Each client maintains a global model (updated via FedAvg) and a separate
    personalized model trained with a proximal regularization term anchored to
    the global model. Evaluation and saving use the personalized model.

    Reference: Li et al., "Ditto: Fair and Robust Federated Learning Through
    Personalization", ICML 2021. arXiv 2012.04235.
    """

    optional = {
        "mu": 0.1,
        "plocal_epochs": 1,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--mu", type=float, default=None)
        parser.add_argument("--plocal_epochs", type=int, default=None)


class Ditto_Client(pFL_Client):
    """
    Client for Ditto. Maintains two models:
    - model: global model trained with standard FedAvg objective
    - model_per: personalized model trained with proximal regularization to model

    Per-client persistent state stored in personal_model_params:
    - "model_per": state dict of the personalized model
    """

    def __init__(self, configs, times, device):
        super().__init__(configs, times, device)
        self.model_per = copy.deepcopy(self.model)
        self._global_params = []

    def set_parameters(self, package: Dict[str, Any]) -> None:
        super().set_parameters(package)
        if "model_per" in package["personal_model_params"]:
            self.model_per.load_state_dict(
                package["personal_model_params"]["model_per"]
            )
        # Capture global model params for proximal term used in fit()
        self._global_params = [p.data.clone() for p in self.model.parameters()]

    def fit(self) -> None:
        # Step 1: train model_per with proximal regularization anchored to global model
        self._set_worker_seed(self._loader_seed("train"))
        loader = self.load_train_data()
        global_params = [g.to(self.device) for g in self._global_params]
        self.model_per.to(self.device)
        self.model_per.train()
        per_optim = torch.optim.SGD(self.model_per.parameters(), lr=self.learning_rate)
        offload_per = self.efficiency == "low"
        for _ in range(self.plocal_epochs):
            for batch_x, batch_y, x_mark, y_mark in loader:
                per_optim.zero_grad()
                batch_x = batch_x.to(device=self.device, dtype=torch.float32)
                batch_y = batch_y.to(device=self.device, dtype=torch.float32)
                x_mark = x_mark.to(device=self.device, dtype=torch.float32)
                y_mark = y_mark.to(device=self.device, dtype=torch.float32)
                outputs = self.model_per(batch_x, x_mark=x_mark, y_mark=y_mark)
                loss = self.loss(outputs, batch_y)
                loss.backward()
                for p, g in zip(self.model_per.parameters(), global_params):
                    if p.grad is not None:
                        p.grad.data.add_(self.mu * (p.data - g.data))
                per_optim.step()
        if offload_per:
            self.model_per.to("cpu")
        del global_params

        # Step 2: train global model with standard objective
        super().fit()

        if self.efficiency == "med":
            self.model_per.to("cpu")

    def package(self, train_time: float) -> Dict[str, Any]:
        result = super().package(train_time)
        result["personal_model_params"]["model_per"] = {
            k: v.detach().cpu().clone()
            for k, v in self.model_per.state_dict().items()
        }
        return result

    def evaluate_personalized(
        self, client_id, global_params, personal_params, dataset_type, current_iter
    ) -> float:
        self.id = client_id
        self.current_iter = current_iter
        self._load_private(client_id)
        self.model.load_state_dict(global_params, strict=False)
        if "model_per" in personal_params:
            self.model.load_state_dict(personal_params["model_per"], strict=False)
        loader = (
            self.load_test_data()
            if dataset_type == "test"
            else self.load_train_data()
        )
        losses = self.calculate_loss(
            model=self.model,
            dataloader=loader,
            criterion=self.loss,
            device=self.device,
            offload_after=self.efficiency != "high",
        )
        return float(np.mean(losses))
