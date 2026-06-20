import copy
from argparse import Namespace
from collections import OrderedDict
from typing import Any, Dict, List

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
    on the envelope.  The server aggregates local models (w_i) and does a
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

    def aggregate_client_updates(self, packages: "OrderedDict[int, dict]") -> None:
        prev_global = copy.deepcopy(self.public_model_params)

        scores = [p["score"] for p in packages.values()]
        total = float(sum(scores))
        weights = torch.tensor([s / total for s in scores], dtype=torch.float32)
        new_params = OrderedDict()
        for name in self.public_model_params:
            stacked = torch.stack(
                [p["regular_model_params"][name] for p in packages.values()], dim=-1
            )
            new_params[name] = torch.sum(stacked * weights.to(stacked.dtype), dim=-1)

        # β-blend: global = (1-β)*prev + β*aggregated
        if self.beta < 1.0:
            for name in new_params:
                new_params[name] = (
                    (1.0 - self.beta) * prev_global[name]
                    + self.beta * new_params[name]
                )

        self._commit_global(new_params)


class pFedMe_Client(pFL_Client):
    """
    Client for pFedMe.

    Maintains personalized params θ*_i (stored in clients_personal_model_params
    under key "personalized_params").  Per round:
      - K inner SGD steps per batch to find θ*_i (Moreau envelope minimizer)
      - Outer update: w_i -= λ*lr*(w_i - θ*_i) (gradient of envelope w.r.t. w)
      - Send w_i to server (model reset to w_i for aggregation)
    """

    def set_parameters(self, package: Dict[str, Any]) -> None:
        super().set_parameters(package)
        pp_list = package["personal_model_params"].get("personalized_params", None)
        if pp_list is None:
            self._personalized_params: List[torch.Tensor] = [
                p.data.clone().cpu() for p in self.model.parameters()
            ]
        else:
            self._personalized_params = [pp.clone() for pp in pp_list]

    def fit(self) -> None:
        self._set_worker_seed(self._loader_seed("train"))
        loader = self.load_train_data()

        self.model.to(self.device)
        self.model.train()

        inner_optim = pFedMeOptimizer(
            self.model.parameters(), lr=self.p_lr, lamda=self.lamda
        )
        # w: local params evolving via outer update (start = server global)
        local_dev = [p.data.clone() for p in self.model.parameters()]

        for _ in range(self.epochs):
            for batch_x, batch_y, x_mark, y_mark in loader:
                batch_x = batch_x.to(device=self.device, dtype=torch.float32)
                batch_y = batch_y.to(device=self.device, dtype=torch.float32)
                x_mark = x_mark.to(device=self.device, dtype=torch.float32)
                y_mark = y_mark.to(device=self.device, dtype=torch.float32)

                # K inner steps: update model params (θ) toward Moreau minimizer
                # minimizes L(θ) + λ/2 * ||θ - w||²
                for _ in range(self.K):
                    inner_optim.zero_grad()
                    outputs = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
                    loss = self.loss(outputs, batch_y)
                    loss.backward()
                    inner_optim.step(local_dev, self.device)

                # Outer update: w -= λ*p_lr*(w - θ*)
                for lp, param in zip(local_dev, self.model.parameters()):
                    lp.data = lp.data - self.lamda * self.p_lr * (lp.data - param.data)

        # θ* = model params after last inner optimization; save before reset
        self._personalized_params = [
            p.data.clone().cpu() for p in self.model.parameters()
        ]

        # Reset model to w (what the server aggregates)
        for param, lp in zip(self.model.parameters(), local_dev):
            param.data.copy_(lp)

        if self.efficiency != "high":
            self.model.to("cpu")

    def package(self, train_time: float) -> Dict[str, Any]:
        out = super().package(train_time)
        out["personal_model_params"]["personalized_params"] = self._personalized_params
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
        self._load_private(client_id)
        self.model.load_state_dict(global_params, strict=False)
        pp_list = personal_params.get("personalized_params", None) if personal_params else None
        if pp_list is not None:
            with torch.no_grad():
                for param, pp in zip(self.model.parameters(), pp_list):
                    param.data.copy_(pp)
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
