from argparse import Namespace
from collections import OrderedDict
from typing import Any, Dict, List

import torch
from torch.optim import Optimizer

from .tFL import tFL, tFL_Client


class NovaOptimizer(Optimizer):
    """SGD optimizer that accumulates the normalized gradient update for FedNova.

    Tracks `cum_grad = sum(lr * g_t)` across all local steps and
    `local_normalizing_vec` (= number of steps for vanilla SGD, adjusted for
    momentum / proximal variants).  The normalized gradient returned to the
    server is `cum_grad / local_normalizing_vec`.
    """

    def __init__(
        self,
        params,
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        prox_mu: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.prox_mu = prox_mu
        self.momentum = momentum
        self.local_normalizing_vec = 0.0
        self.local_counter = 0.0
        self.local_steps = 0

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad.data.clone()

                if wd != 0:
                    d_p = d_p.add(p.data, alpha=wd)

                param_state = self.state[p]

                # Save initial params for proximal term
                if "old_init" not in param_state:
                    param_state["old_init"] = p.data.clone().detach()

                # Momentum buffer
                if mu != 0:
                    if "momentum_buffer" not in param_state:
                        param_state["momentum_buffer"] = d_p.clone()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(mu).add_(d_p, alpha=1.0 - 0.0)
                        d_p = buf

                # Proximal correction
                if self.prox_mu != 0:
                    d_p = d_p.add(p.data - param_state["old_init"], alpha=self.prox_mu)

                # Accumulate gradient for normalizing
                if "cum_grad" not in param_state:
                    param_state["cum_grad"] = d_p.clone().mul_(lr)
                else:
                    param_state["cum_grad"].add_(d_p, alpha=lr)

                p.data.add_(d_p, alpha=-lr)

        # Update normalizing vector
        if self.momentum != 0:
            self.local_counter = self.local_counter * self.momentum + 1.0
            self.local_normalizing_vec += self.local_counter
        if self.prox_mu != 0:
            etamu = group["lr"] * self.prox_mu
            self.local_normalizing_vec *= 1.0 - etamu
            self.local_normalizing_vec += 1.0
        if self.momentum == 0 and self.prox_mu == 0:
            self.local_normalizing_vec += 1.0

        self.local_steps += 1
        return loss


class FedNova(tFL):
    """
    FedNova: Tackling the Objective Inconsistency Problem in Heterogeneous
    Federated Optimization.

    Each client normalizes its gradient update by the local normalizing
    vector a_i (= number of local steps for vanilla SGD) before sending it
    to the server.  The server aggregates:
        Δ = τ_eff · weighted_avg(d_i)
        global = global - Δ
    where τ_eff = weighted_avg(τ_i) and d_i = cum_grad_i / a_i.

    Optional global momentum (gmf) can be enabled to further smooth the
    server update.

    Reference: Wang et al., "Tackling the Objective Inconsistency Problem
    in Heterogeneous Federated Optimization", NeurIPS 2020. arXiv 2007.07481.
    """

    optional = {
        "gmf": 0.0,
        "prox_mu": 0.0,
        "nova_momentum": 0.0,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--gmf", type=float, default=None)
        parser.add_argument("--prox_mu", type=float, default=None)
        parser.add_argument("--nova_momentum", type=float, default=None)

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        self.parallel = False
        self._global_momentum_buffer: List[torch.Tensor] = []

    def aggregate_client_updates(self, packages: "OrderedDict[int, dict]") -> None:
        total_score = sum(pkg["score"] for pkg in packages.values())

        # τ_eff = weighted_avg(tau_i)
        tau_eff = sum(
            pkg["tau"] * (pkg["score"] / total_score) for pkg in packages.values()
        )

        # weighted_avg(d_i) where d_i = cum_grad_i / a_i
        avg_d: List[torch.Tensor] = [
            sum(
                pkg["nova_grad"][i] * (pkg["score"] / total_score)
                for pkg in packages.values()
            )
            for i in range(len(self.public_model_params))
        ]

        # Global momentum
        if self.gmf != 0.0:
            if not self._global_momentum_buffer:
                self._global_momentum_buffer = [
                    (tau_eff * d / self.learning_rate).clone() for d in avg_d
                ]
            else:
                for buf, d in zip(self._global_momentum_buffer, avg_d):
                    buf.mul_(self.gmf).add_(tau_eff * d / self.learning_rate)
            update = [self.learning_rate * buf for buf in self._global_momentum_buffer]
        else:
            update = [tau_eff * d for d in avg_d]

        new_params = OrderedDict(
            (name, param - upd.to(param.device))
            for (name, param), upd in zip(self.public_model_params.items(), update)
        )
        self._commit_global(new_params)


class FedNova_Client(tFL_Client):
    """
    Client for FedNova. Uses NovaOptimizer to accumulate normalized gradient
    updates and returns nova_grad = cum_grad / local_normalizing_vec and tau.
    """

    personal_params_name: List[str] = []

    def fit(self) -> None:
        train_loader = self.load_train_data()

        nova_opt = NovaOptimizer(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.nova_momentum,
            prox_mu=self.prox_mu,
        )

        self.model.to(self.device)
        self.model.train()

        for _ in range(self.epochs):
            for batch_x, batch_y, x_mark, y_mark in train_loader:
                nova_opt.zero_grad()
                batch_x = batch_x.to(device=self.device, dtype=torch.float32)
                batch_y = batch_y.to(device=self.device, dtype=torch.float32)
                x_mark = x_mark.to(device=self.device, dtype=torch.float32)
                y_mark = y_mark.to(device=self.device, dtype=torch.float32)
                outputs = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
                loss = self.loss(outputs, batch_y)
                loss.backward()
                nova_opt.step()

        # Collect normalized gradient: d_i = cum_grad / a_i
        a_i = (
            nova_opt.local_normalizing_vec
            if nova_opt.local_normalizing_vec > 0
            else 1.0
        )
        nova_grad = []
        for p in self.model.parameters():
            state = nova_opt.state[p]
            if "cum_grad" in state:
                nova_grad.append(state["cum_grad"].cpu() / a_i)
            else:
                nova_grad.append(torch.zeros_like(p, device="cpu"))

        self._nova_grad = nova_grad
        self._tau = nova_opt.local_normalizing_vec
        self.model.to("cpu")

    def package(self, train_time: float) -> Dict[str, Any]:
        result = super().package(train_time)
        result["nova_grad"] = self._nova_grad
        result["tau"] = self._tau
        return result
