import copy
import time
from argparse import Namespace
from typing import Any, Dict, List

import torch
from torch.optim import Optimizer

from .tFL import tFL, tFL_Client


class SCAFFOLDOptimizer(Optimizer):
    """SGD with SCAFFOLD control-variate correction: p -= lr * (grad + c_server - c_client)."""

    def __init__(self, params, lr: float) -> None:
        super().__init__(params, dict(lr=lr))

    def step(
        self, server_cs: List[torch.Tensor], client_cs: List[torch.Tensor]
    ) -> None:
        for group in self.param_groups:
            for p, sc, cc in zip(group["params"], server_cs, client_cs):
                if p.grad is None:
                    continue
                correction = sc.to(p.device) - cc.to(p.device)
                p.data.add_(p.grad.data + correction, alpha=-group["lr"])


class SCAFFOLD(tFL):
    """
    SCAFFOLD: Stochastic Controlled Averaging for Federated Learning.

    Maintains server-side and per-client control variates to correct for
    client drift caused by heterogeneous local data distributions.

    Reference: Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging
    for Federated Learning", ICML 2020. arXiv 1910.06378.
    """

    optional = {
        "server_lr": 1.0,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--server_lr", type=float, default=None)

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        # SCAFFOLD needs delta_c propagation from workers → disable Ray parallelism
        self.parallel = False
        self.global_c: List[torch.Tensor] = [
            torch.zeros_like(p, device="cpu") for p in self.model.parameters()
        ]

    def variables_to_be_sent(self) -> Dict[str, Any]:
        self._round_snapshot = copy.deepcopy(self.model)
        return {"model": self.model, "global_c": self.global_c}

    def aggregate_models(self) -> None:
        K = len(self.client_data)
        snapshot_params = list(self._round_snapshot.parameters())

        # Global model update: theta += server_lr / K * sum(theta_local - theta_global)
        for i, param in enumerate(self.model.parameters()):
            delta_sum = sum(
                list(cd["model"].parameters())[i].data.to(param.device)
                - snapshot_params[i].data
                for cd in self.client_data
            )
            param.data.add_(delta_sum / K, alpha=self.server_lr)

        # Control variate update: c += (1/N) * sum(delta_c)
        N = self.num_clients
        for i, gc in enumerate(self.global_c):
            delta_sum = sum(cd["delta_c"][i].to(gc.device) for cd in self.client_data)
            gc.data.add_(delta_sum / N)

        del self._round_snapshot


class SCAFFOLD_Client(tFL_Client):
    """
    Client for SCAFFOLD. Uses SCAFFOLDOptimizer and maintains local control
    variates c_i. Returns updated model + delta_c after each round.
    """

    def __init__(self, configs: Namespace, id: int, times: int) -> None:
        super().__init__(configs=configs, id=id, times=times)
        self.client_c: List[torch.Tensor] = [
            torch.zeros_like(p, device="cpu") for p in self.model.parameters()
        ]
        self._global_c: List[torch.Tensor] = [
            torch.zeros_like(p, device="cpu") for p in self.model.parameters()
        ]
        self._global_snapshot: List[torch.Tensor] = []
        self.delta_c: List[torch.Tensor] = []

    def receive_from_server(self, data: dict) -> None:
        if "global_c" in data:
            self._global_c = [c.clone().cpu() for c in data["global_c"]]
        self._global_snapshot = [
            p.data.clone().cpu() for p in data["model"].parameters()
        ]
        self.update_model_params(old=self.model, new=data["model"])

    def train(self):
        train_loader = self.load_train_data()
        start_time = time.time()

        scaffold_optim = SCAFFOLDOptimizer(
            self.model.parameters(), lr=self.learning_rate
        )
        self.model.to(self.device)
        self.model.train()

        num_steps = 0
        for _ in range(self.epochs):
            for batch_x, batch_y, x_mark, y_mark in train_loader:
                scaffold_optim.zero_grad()
                batch_x = batch_x.to(device=self.device, dtype=torch.float32)
                batch_y = batch_y.to(device=self.device, dtype=torch.float32)
                x_mark = x_mark.to(device=self.device, dtype=torch.float32)
                y_mark = y_mark.to(device=self.device, dtype=torch.float32)
                outputs = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
                loss = self.loss(outputs, batch_y)
                loss.backward()
                scaffold_optim.step(self._global_c, self.client_c)
                num_steps += 1

        # Compute delta_c and update client_c
        # delta_c = -global_c + (1 / (num_steps * lr)) * (theta_global_before - theta_local)
        delta_c = []
        new_client_c = []
        inv_lr_steps = 1.0 / (num_steps * self.learning_rate) if num_steps > 0 else 0.0
        for gc, cc, g_snap, lp in zip(
            self._global_c,
            self.client_c,
            self._global_snapshot,
            self.model.parameters(),
        ):
            dc = -gc + inv_lr_steps * (g_snap - lp.detach().cpu())
            delta_c.append(dc)
            new_client_c.append(cc + dc)

        self.delta_c = delta_c
        self.client_c = new_client_c
        self.model.to("cpu")
        self.metrics["train_time"].append(time.time() - start_time)

    def variables_to_be_sent(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "delta_c": self.delta_c,
            "score": self.train_samples,
        }

    def send_to_server(self) -> Dict[str, Any]:
        to_be_sent = self.variables_to_be_sent()
        model_size = self.get_size(to_be_sent["model"])
        self.metrics["send_mb"].append(model_size)
        return to_be_sent
