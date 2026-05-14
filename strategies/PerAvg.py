import copy
import time
from argparse import Namespace

import torch

from .tFL import tFL, tFL_Client


class PerAvg(tFL):
    """
    Per-FedAvg: Personalized Federated Averaging via Model-Agnostic Meta-Learning.

    Trains a global initialization θ such that one local gradient step from θ
    yields a good personalized model for each client (first-order MAML).
    Server aggregates global models via FedAvg.

    Reference: Fallah et al., "Personalized Federated Learning with Theoretical
    Guarantees: A Model-Agnostic Meta-Learning Approach", NeurIPS 2020.
    arXiv 2002.07948.
    """

    optional = {
        "beta": 1e-3,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--beta", type=float, default=None)

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        self.parallel = False


class PerAvg_Client(tFL_Client):
    """
    Client for PerAvg. Uses a double-batch MAML update:
    1. Take a temporary inner step α on the first half-batch.
    2. Compute gradient from the second half-batch at the updated params.
    3. Restore original params and apply outer step β using that gradient.
    """

    def train(self):
        # Load double-sized batches so we can split 50/50
        train_loader = self.load_train_data()
        start_time = time.time()

        self.model.to(self.device)
        self.model.train()

        for _ in range(self.epochs):
            for batch_x, batch_y, x_mark, y_mark in train_loader:
                batch_x = batch_x.to(device=self.device, dtype=torch.float32)
                batch_y = batch_y.to(device=self.device, dtype=torch.float32)
                x_mark = x_mark.to(device=self.device, dtype=torch.float32)
                y_mark = y_mark.to(device=self.device, dtype=torch.float32)

                half = batch_x.size(0) // 2
                if half == 0:
                    half = batch_x.size(0)

                # Save current params
                temp_params = [p.data.clone() for p in self.model.parameters()]

                # Inner step α on first half
                self.optimizer.zero_grad()
                out = self.model(batch_x[:half], x_mark=x_mark[:half], y_mark=y_mark[:half])
                loss = self.loss(out, batch_y[:half])
                loss.backward()
                with torch.no_grad():
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.data.sub_(self.learning_rate * p.grad)

                # Compute gradient from second half at updated params
                self.optimizer.zero_grad()
                out2 = self.model(
                    batch_x[half:] if half < batch_x.size(0) else batch_x,
                    x_mark=x_mark[half:] if half < batch_x.size(0) else x_mark,
                    y_mark=y_mark[half:] if half < batch_x.size(0) else y_mark,
                )
                loss2 = self.loss(
                    out2,
                    batch_y[half:] if half < batch_x.size(0) else batch_y,
                )
                loss2.backward()

                # Restore params and apply outer step β
                with torch.no_grad():
                    for p, tp in zip(self.model.parameters(), temp_params):
                        p.data.copy_(tp)
                        if p.grad is not None:
                            p.data.sub_(self.beta * p.grad)

        self.scheduler.step()
        if self.efficiency != "high":
            self.model.to("cpu")
        self.metrics["train_time"].append(time.time() - start_time)
