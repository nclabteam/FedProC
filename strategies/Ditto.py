import copy
import time
from argparse import Namespace

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

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        # Ditto trains two models per client; disable Ray parallelism
        self.parallel = False

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--mu", type=float, default=None)
        parser.add_argument("--plocal_epochs", type=int, default=None)

    def save_models(self, save_type: str) -> None:
        if save_type not in ["last", "best"]:
            raise ValueError("save_type must be 'last' or 'best'")

        should_save = True
        if save_type == "best":
            metric_key = "personal_avg_test_loss"
            if metric_key not in self.metrics or not self.metrics[metric_key]:
                should_save = False
            else:
                vals = self.metrics[metric_key]
                if vals[-1] != min(vals):
                    should_save = False

        if not should_save:
            return

        if not self.exclude_server_model_processes:
            self.save_model(
                model=self.model,
                path=self.model_path,
                name=self.name,
                postfix=save_type,
                configs=self.configs,
                metadata={"save_type": save_type, "owner": "server"},
                verbose=self.logger,
            )

        for client in self.clients:
            client.save_model(
                model=client.model_per,
                path=client.model_path,
                name=client.name,
                postfix=save_type,
                configs=client.configs,
                metadata={"save_type": save_type, "owner": client.name},
                verbose=client.logger,
            )


class Ditto_Client(pFL_Client):
    """
    Client for Ditto. Maintains two models:
    - model: global model trained with standard FedAvg objective
    - model_per: personalized model trained with proximal regularization to model
    """

    def __init__(self, configs: Namespace, id: int, times: int) -> None:
        super().__init__(configs=configs, id=id, times=times)
        self.model_per = copy.deepcopy(self.model)
        self._global_snapshot = None

    def receive_from_server(self, data: dict) -> None:
        self._global_snapshot = copy.deepcopy(data["model"]).to("cpu")
        self.update_model_params(old=self.model, new=data["model"])

    def train(self):
        train_loader = self.load_train_data()
        start_time = time.time()

        # Step 1: train personalized model with proximal regularization to global model
        global_params = list(self._global_snapshot.to(self.device).parameters())
        self.model_per.to(self.device)
        self.model_per.train()
        per_optim = torch.optim.SGD(self.model_per.parameters(), lr=self.learning_rate)
        offload_per = self.efficiency == "low"
        for _ in range(self.plocal_epochs):
            for batch_x, batch_y, x_mark, y_mark in train_loader:
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
        offload_after = self.efficiency == "low"
        for _ in range(self.epochs):
            self.train_one_epoch(
                model=self.model,
                dataloader=train_loader,
                optimizer=self.optimizer,
                criterion=self.loss,
                scheduler=self.scheduler,
                device=self.device,
                offload_after=offload_after,
            )
        if self.efficiency == "med":
            self.model.to("cpu")
            self.model_per.to("cpu")

        self.metrics["train_time"].append(time.time() - start_time)

    def get_train_loss(self) -> float:
        losses = self.calculate_loss(
            model=self.model_per,
            dataloader=self.load_train_data(),
            criterion=self.loss,
            device=self.device,
            offload_after=self.efficiency != "high",
        )
        loss = float(np.mean(losses))
        self.metrics["train_loss"].append(loss)
        return loss

    def get_test_loss(self) -> float:
        losses = self.calculate_loss(
            model=self.model_per,
            dataloader=self.load_test_data(),
            criterion=self.loss,
            device=self.device,
            offload_after=self.efficiency != "high",
        )
        loss = float(np.mean(losses))
        self.metrics["test_loss"].append(loss)
        return loss
