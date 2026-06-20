import copy
from argparse import Namespace
from collections import OrderedDict
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

    def package(self, client_id: int) -> Dict[str, Any]:
        pkg = super().package(client_id)
        pkg["global_c"] = copy.deepcopy(self.global_c)
        return pkg

    def aggregate_client_updates(self, packages: "OrderedDict[int, dict]") -> None:
        K = len(packages)
        # Snapshot of global params at the moment clients received them
        snapshot = copy.deepcopy(self.public_model_params)

        # Global model update: theta += server_lr / K * sum(theta_local - theta_global)
        new_params = OrderedDict()
        for name, snap_val in snapshot.items():
            delta_sum = sum(
                pkg["regular_model_params"][name].to(snap_val.device) - snap_val
                for pkg in packages.values()
            )
            new_params[name] = snap_val + self.server_lr * delta_sum / K

        # Control variate update: c += (1/N) * sum(delta_c)
        N = self.num_clients
        for i, gc in enumerate(self.global_c):
            delta_sum = sum(pkg["delta_c"][i].to(gc.device) for pkg in packages.values())
            gc.data.add_(delta_sum / N)

        self._commit_global(new_params)


class SCAFFOLD_Client(tFL_Client):
    """
    Client for SCAFFOLD. Uses SCAFFOLDOptimizer and maintains local control
    variates c_i. Returns updated model + delta_c after each round.
    """

    client_c = None  # initialized lazily in set_parameters on first round

    def set_parameters(self, package: Dict[str, Any]) -> None:
        super().set_parameters(package)
        client_c = package["personal_model_params"].get("client_c", None)
        if client_c is None:
            self.client_c = [
                torch.zeros_like(p, device="cpu") for p in self.model.parameters()
            ]
        else:
            self.client_c = client_c
        self.global_c = package["global_c"]
        self._global_snapshot = [
            v.clone().cpu() for v in package["regular_model_params"].values()
        ]

    def fit(self) -> None:
        train_loader = self.load_train_data()

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
                scaffold_optim.step(self.global_c, self.client_c)
                num_steps += 1

        # Compute delta_c and update client_c
        # delta_c = -global_c + (1 / (num_steps * lr)) * (theta_global_before - theta_local)
        delta_c = []
        new_client_c = []
        inv_lr_steps = 1.0 / (num_steps * self.learning_rate) if num_steps > 0 else 0.0
        for gc, cc, g_snap, lp in zip(
            self.global_c,
            self.client_c,
            self._global_snapshot,
            self.model.parameters(),
        ):
            dc = -gc + inv_lr_steps * (g_snap - lp.detach().cpu())
            delta_c.append(dc)
            new_client_c.append(cc + dc)

        self.client_c = new_client_c
        self._delta_c = delta_c
        self.model.to("cpu")

    def package(self, train_time: float) -> Dict[str, Any]:
        result = super().package(train_time)
        result["personal_model_params"]["client_c"] = self.client_c
        result["delta_c"] = self._delta_c
        return result
