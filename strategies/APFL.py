import copy
import time
from argparse import Namespace

import numpy as np
import torch

from .pFL import pFL, pFL_Client


class APFL(pFL):
    """
    APFL: Adaptive Personalized Federated Learning.

    Each client maintains a global model w (aggregated via FedAvg) and a
    personalized model v_i.  Per round, both models are trained simultaneously
    and their outputs mixed with a per-client adaptive coefficient α_i.

    Reference: Deng et al., "Adaptive Personalized Federated Learning",
    arXiv 2003.13461.
    """

    optional = {
        "alpha": 0.5,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--alpha", type=float, default=None)

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        self.parallel = False

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


class APFL_Client(pFL_Client):
    """
    Client for APFL. Trains global model w (sent to server) and personalized
    model v_i (kept local) simultaneously, updating mixing coefficient α_i
    adaptively each batch.
    """

    def __init__(self, configs: Namespace, id: int, times: int) -> None:
        super().__init__(configs=configs, id=id, times=times)
        self.model_per = copy.deepcopy(self.model)
        self.alpha: float = self.alpha  # per-client mixing coefficient

    def receive_from_server(self, data: dict) -> None:
        self.update_model_params(old=self.model, new=data["model"])

    def train(self):
        train_loader = self.load_train_data()
        start_time = time.time()

        optim_g = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        optim_p = torch.optim.SGD(self.model_per.parameters(), lr=self.learning_rate)

        self.model.to(self.device)
        self.model_per.to(self.device)
        self.model.train()
        self.model_per.train()

        for _ in range(self.epochs):
            for batch_x, batch_y, x_mark, y_mark in train_loader:
                batch_x = batch_x.to(device=self.device, dtype=torch.float32)
                batch_y = batch_y.to(device=self.device, dtype=torch.float32)
                x_mark = x_mark.to(device=self.device, dtype=torch.float32)
                y_mark = y_mark.to(device=self.device, dtype=torch.float32)

                # Train global model
                optim_g.zero_grad()
                out_g = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
                loss_g = self.loss(out_g, batch_y)
                loss_g.backward()
                optim_g.step()

                # Train personalized model
                optim_p.zero_grad()
                out_p = self.model_per(batch_x, x_mark=x_mark, y_mark=y_mark)
                loss_p = self.loss(out_p, batch_y)
                loss_p.backward()
                optim_p.step()

                # Adaptive alpha update
                self._update_alpha()

        # Mix: v_i = (1-α)*w + α*v_i
        for lp, gp in zip(self.model_per.parameters(), self.model.parameters()):
            lp.data = (1.0 - self.alpha) * gp.data + self.alpha * lp.data

        self.model.to("cpu")
        self.model_per.to("cpu")
        self.metrics["train_time"].append(time.time() - start_time)

    def _update_alpha(self) -> None:
        grad_alpha = 0.0
        for gp, pp in zip(self.model.parameters(), self.model_per.parameters()):
            if pp.grad is None or gp.grad is None:
                continue
            dif = (pp.data - gp.data).view(-1)
            grad = (self.alpha * pp.grad.data + (1.0 - self.alpha) * gp.grad.data).view(
                -1
            )
            grad_alpha += float(dif.dot(grad))
        grad_alpha += 0.02 * self.alpha
        self.alpha = float(
            np.clip(self.alpha - self.learning_rate * grad_alpha, 0.0, 1.0)
        )

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
