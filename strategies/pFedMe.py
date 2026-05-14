import copy
import time
from argparse import Namespace
from typing import List

import numpy as np
import torch
from torch.optim import Optimizer

from .pFL import pFL, pFL_Client


class pFedMeOptimizer(Optimizer):
    """
    Inner optimizer for pFedMe: minimizes the Moreau envelope surrogate
    f(θ) + λ/2 * ||θ - w||² w.r.t. θ (holding local params w fixed).
    """

    def __init__(self, params, lr: float = 0.01, lamda: float = 0.1) -> None:
        super().__init__(params, dict(lr=lr, lamda=lamda))

    def step(self, local_params: List[torch.Tensor], device: str):
        for group in self.param_groups:
            for p, lw in zip(group["params"], local_params):
                if p.grad is None:
                    continue
                lw = lw.to(device)
                p.data = p.data - group["lr"] * (
                    p.grad.data + group["lamda"] * (p.data - lw.data)
                )
        return [p for group in self.param_groups for p in group["params"]]


class pFedMe(pFL):
    """
    pFedMe: Personalized Federated Learning with Moreau Envelopes.

    Each client finds a personalized model θ*_i that minimizes the Moreau
    envelope objective, then updates a local model w_i via gradient descent
    on the envelope.  The server aggregates local models and does a
    β-weighted blend with the previous global model.

    Reference: Dinh et al., "Personalized Federated Learning with Moreau
    Envelopes", NeurIPS 2020. arXiv 2006.08848.
    """

    optional = {
        "lamda": 15.0,
        "K": 5,
        "p_lr": 0.01,
        "beta": 1.0,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--lamda", type=float, default=None)
        parser.add_argument("--K", type=int, default=None)
        parser.add_argument("--p_lr", type=float, default=None)
        parser.add_argument("--beta", type=float, default=None)

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        self.parallel = False
        self._prev_global_params: List[torch.Tensor] = []

    def aggregate_models(self) -> None:
        # Save global model before standard aggregation for beta-blend
        self._prev_global_params = [
            p.data.clone() for p in self.model.parameters()
        ]
        super().aggregate_models()
        # β-blend: global = (1-β)*prev + β*aggregated
        if self.beta < 1.0:
            for prev, param in zip(self._prev_global_params, self.model.parameters()):
                param.data = (1.0 - self.beta) * prev + self.beta * param.data


class pFedMe_Client(pFL_Client):
    """
    Client for pFedMe. Maintains local params w_i and personalized params θ*_i.
    Per round:
    - K inner SGD steps per batch to find θ*_i (Moreau envelope minimizer)
    - Outer update: w_i -= λ*lr*(w_i - θ*_i) (gradient of envelope w.r.t. w)
    - Send w_i to server (reset model to w_i for aggregation)
    """

    def __init__(self, configs: Namespace, id: int, times: int) -> None:
        super().__init__(configs=configs, id=id, times=times)
        self.local_params: List[torch.Tensor] = [
            p.data.clone() for p in self.model.parameters()
        ]
        self.personalized_params: List[torch.Tensor] = [
            p.data.clone() for p in self.model.parameters()
        ]

    def receive_from_server(self, data: dict) -> None:
        self.update_model_params(old=self.model, new=data["model"])
        for lp, gp in zip(self.local_params, data["model"].parameters()):
            lp.data.copy_(gp.data)

    def train(self):
        train_loader = self.load_train_data()
        start_time = time.time()

        inner_optim = pFedMeOptimizer(
            self.model.parameters(), lr=self.p_lr, lamda=self.lamda
        )
        self.model.to(self.device)
        self.model.train()
        local_dev = [lp.to(self.device) for lp in self.local_params]

        for _ in range(self.epochs):
            for batch_x, batch_y, x_mark, y_mark in train_loader:
                batch_x = batch_x.to(device=self.device, dtype=torch.float32)
                batch_y = batch_y.to(device=self.device, dtype=torch.float32)
                x_mark = x_mark.to(device=self.device, dtype=torch.float32)
                y_mark = y_mark.to(device=self.device, dtype=torch.float32)

                # K inner steps → find personalized θ*_i for this batch
                for _ in range(self.K):
                    inner_optim.zero_grad()
                    outputs = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
                    loss = self.loss(outputs, batch_y)
                    loss.backward()
                    self.personalized_params = inner_optim.step(local_dev, self.device)

                # Outer update of local params: w -= λ*lr*(w - θ*)
                for lp, pp in zip(local_dev, self.personalized_params):
                    lp.data = lp.data - self.lamda * self.p_lr * (lp.data - pp.data)

        # Store updated local params and reset model to w_i for aggregation
        self.local_params = [lp.detach().cpu() for lp in local_dev]
        self.personalized_params = [pp.detach().cpu() for pp in self.personalized_params]
        for param, lp in zip(self.model.parameters(), self.local_params):
            param.data.copy_(lp)
        self.model.to("cpu")
        self.metrics["train_time"].append(time.time() - start_time)

    def get_train_loss(self) -> float:
        # Temporarily load personalized params into model for evaluation
        saved = [p.data.clone() for p in self.model.parameters()]
        for param, pp in zip(self.model.parameters(), self.personalized_params):
            param.data.copy_(pp)
        losses = self.calculate_loss(
            model=self.model,
            dataloader=self.load_train_data(),
            criterion=self.loss,
            device=self.device,
            offload_after=False,
        )
        for param, s in zip(self.model.parameters(), saved):
            param.data.copy_(s)
        loss = float(np.mean(losses))
        self.metrics["train_loss"].append(loss)
        return loss

    def get_test_loss(self) -> float:
        saved = [p.data.clone() for p in self.model.parameters()]
        for param, pp in zip(self.model.parameters(), self.personalized_params):
            param.data.copy_(pp)
        losses = self.calculate_loss(
            model=self.model,
            dataloader=self.load_test_data(),
            criterion=self.loss,
            device=self.device,
            offload_after=False,
        )
        for param, s in zip(self.model.parameters(), saved):
            param.data.copy_(s)
        self.model.to("cpu")
        loss = float(np.mean(losses))
        self.metrics["test_loss"].append(loss)
        return loss
