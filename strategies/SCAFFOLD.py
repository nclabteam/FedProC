import copy
from argparse import Namespace
from collections import OrderedDict
from typing import Any, Dict, List

import torch
from torch.optim import Optimizer

from .tFL import tFL, tFL_Client


class SCAFFOLDOptimizer(Optimizer):
    """SGD with SCAFFOLD control-variate correction: p -= lr * (grad + c_server - c_client)."""

    def __init__(self, params: Any, lr: float) -> None:
        super().__init__(params=params, defaults=dict(lr=lr))

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
    """SCAFFOLD: Stochastic Controlled Averaging for Federated Learning."""

    optional = {
        "server_lr": 1.0,
    }

    @classmethod
    def args_update(cls, parser: Any) -> None:
        parser.add_argument("--server_lr", type=float, default=None)

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        self.parallel = False
        self.global_c: List[torch.Tensor] = [
            torch.zeros_like(p, device="cpu") for p in self.model.parameters()
        ]
        zero_c = [torch.zeros_like(p, device="cpu") for p in self.model.parameters()]
        for cid in range(self.num_clients):
            self.clients_personal_model_params[cid]["client_c"] = [
                t.clone() for t in zero_c
            ]

    def package(self, client_id: int) -> Dict[str, Any]:
        pkg = super().package(client_id=client_id)
        pkg["global_c"] = copy.deepcopy(self.global_c)
        pkg["__wire__"] += ("global_c",)
        return pkg

    def aggregate_client_updates(self, packages: "OrderedDict[int, dict]") -> None:
        num_packages = len(packages)
        if not num_packages:
            raise ValueError("SCAFFOLD requires at least one client update")
        # Snapshot of global params at the moment clients received them
        snapshot = copy.deepcopy(self.public_model_params)

        # Global model update: theta += server_lr / K * sum(theta_local - theta_global)
        model_delta_sums = OrderedDict(
            (name, torch.zeros_like(value)) for name, value in snapshot.items()
        )
        control_delta_sums = [torch.zeros_like(value) for value in self.global_c]
        for package in packages.values():
            for name, snapshot_value in snapshot.items():
                model_delta_sums[name].add_(
                    package["regular_model_params"][name].to(snapshot_value.device)
                    - snapshot_value
                )
            for index, global_control in enumerate(self.global_c):
                control_delta_sums[index].add_(
                    package["delta_c"][index].to(global_control.device)
                )
        new_params = OrderedDict(
            (
                name,
                value + self.server_lr * model_delta_sums[name] / num_packages,
            )
            for name, value in snapshot.items()
        )

        # Control variate update: c += (1/N) * sum(delta_c)
        for global_control, delta_sum in zip(self.global_c, control_delta_sums):
            global_control.data.add_(delta_sum / self.num_clients)

        self._commit_global(new_params=new_params)


class SCAFFOLD_Client(tFL_Client):
    """Client for SCAFFOLD."""

    def set_parameters(self, package: Dict[str, Any]) -> None:
        super().set_parameters(package=package)
        self.client_c = package["personal_model_params"]["client_c"]
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

    def package(self) -> Dict[str, Any]:
        result = super().package()
        result["personal_model_params"]["client_c"] = self.client_c
        result["delta_c"] = self._delta_c
        result["__wire__"] += ("delta_c",)
        return result
