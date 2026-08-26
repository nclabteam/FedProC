from argparse import Namespace
from collections import OrderedDict
from typing import Any, Dict, List

import numpy as np
import torch
from torch.optim import Optimizer

from .pFL import pFL, pFL_Client


class pFedMeOptimizer(Optimizer):
    """Inner optimizer for pFedMe: gradient step on f(θ) + λ/2||θ - w||² w.r.t. θ."""

    def __init__(self, params: Any, lr: float = 0.01, lamda: float = 0.1) -> None:
        super().__init__(params=params, defaults=dict(lr=lr, lamda=lamda))

    @torch.no_grad()
    def step(self, local_params: List[torch.Tensor]) -> None:
        for group in self.param_groups:
            for p, lw in zip(group["params"], local_params):
                if p.grad is None:
                    continue
                p.add_(
                    p.grad + group["lamda"] * (p - lw),
                    alpha=-group["lr"],
                )


class pFedMe(pFL):
    """pFedMe: Personalized Federated Learning with Moreau Envelopes (Dinh et al., NeurIPS 2020)."""

    optional = {
        "lamda": 15.0,
        "K": 5,
        "p_lr": 0.09,
        "beta": 1.0,
    }

    @classmethod
    def args_update(cls, parser: Any) -> None:
        parser.add_argument("--lamda", type=float, default=None)
        parser.add_argument("--K", type=int, default=None)
        parser.add_argument("--p_lr", type=float, default=None)
        parser.add_argument("--beta", type=float, default=None)

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        init_pp = [p.data.cpu().clone() for p in self.model.parameters()]
        for cid in range(self.num_clients):
            self.clients_personal_model_params[cid]["personalized_params"] = [
                t.clone() for t in init_pp
            ]

    def aggregate_client_updates(self, packages: "OrderedDict[int, dict]") -> None:
        mean = self.mean_models(
            models=[package["regular_model_params"] for package in packages.values()]
        )
        self._commit_global(
            new_params=OrderedDict(
                (
                    name,
                    (1.0 - self.beta) * self.public_model_params[name]
                    + self.beta * value,
                )
                for name, value in mean.items()
            )
        )

    def select_clients(self) -> None:
        super().select_clients()
        self.aggregation_clients = self.selected_clients
        self.selected_clients = [
            cid for cid in range(self.num_clients) if not self.is_new[cid]
        ]

    def package(self, client_id: int) -> Dict[str, Any]:
        package = super().package(client_id=client_id)
        package["upload_model"] = client_id in self.aggregation_clients
        return package

    def train_one_round(self) -> dict:
        packages = self.trainer.train(self.selected_clients)
        self.aggregate_client_updates(
            packages=OrderedDict(
                (cid, packages[cid]) for cid in self.aggregation_clients
            )
        )
        return packages


class pFedMe_Client(pFL_Client):
    """Client for pFedMe."""

    def set_parameters(self, package: Dict[str, Any]) -> None:
        super().set_parameters(package=package)
        self.upload_model = package["upload_model"]
        self._personalized_params: List[torch.Tensor] = [
            pp.clone() for pp in package["personal_model_params"]["personalized_params"]
        ]

    def fit(self) -> None:
        self._set_worker_seed(seed=self._loader_seed(dataset_type="train"))
        loader = self.load_train_data()

        self.model.to(self.device)
        self.model.train()

        inner_optim = pFedMeOptimizer(
            self.model.parameters(), lr=self.p_lr, lamda=self.lamda
        )
        local_dev = [p.data.clone() for p in self.model.parameters()]

        batches = iter(loader)
        for _ in range(self.epochs):
            try:
                batch_x, batch_y, x_mark, y_mark = next(batches)
            except StopIteration:
                batches = iter(loader)
                batch_x, batch_y, x_mark, y_mark = next(batches)

            batch_x = batch_x.to(device=self.device, dtype=torch.float32)
            batch_y = batch_y.to(device=self.device, dtype=torch.float32)
            x_mark = x_mark.to(device=self.device, dtype=torch.float32)
            y_mark = y_mark.to(device=self.device, dtype=torch.float32)

            for _ in range(self.K):
                inner_optim.zero_grad()
                outputs = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
                loss = self.loss(outputs, batch_y)
                loss.backward()
                inner_optim.step(local_dev)

            with torch.no_grad():
                for lp, param in zip(local_dev, self.model.parameters()):
                    lp.add_(lp - param, alpha=-self.lamda * self.learning_rate)

        self._personalized_params = [
            p.data.clone().cpu() for p in self.model.parameters()
        ]

        with torch.no_grad():
            for param, lp in zip(self.model.parameters(), local_dev):
                param.copy_(lp)

        if self.efficiency != "high":
            self.model.to("cpu")

    def package(self) -> Dict[str, Any]:
        out = super().package()
        out["personal_model_params"]["personalized_params"] = self._personalized_params
        if not self.upload_model:
            out["__wire__"] = ("personal_model_params",)
        return out

    def evaluate_personalized(
        self,
        client_id: int,
        global_params: "OrderedDict[str, torch.Tensor]",
        personal_params: Dict[str, Any],
        dataset_type: str,
        current_iter: int,
    ) -> float:
        self.id = client_id
        self.current_iter = current_iter
        self._load_private(client_id=client_id)
        self.model.load_state_dict(global_params, strict=False)
        with torch.no_grad():
            for param, pp in zip(
                self.model.parameters(), personal_params["personalized_params"]
            ):
                param.data.copy_(pp)
        loader = (
            self.load_test_data() if dataset_type == "test" else self.load_train_data()
        )
        losses = self.calculate_loss(
            model=self.model,
            dataloader=loader,
            criterion=self.loss,
            device=self.device,
            offload_after=self.efficiency != "high",
        )
        return float(np.mean(losses))
