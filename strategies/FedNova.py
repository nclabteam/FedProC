from argparse import Namespace
from collections import OrderedDict
from typing import Any, Dict, List

import torch
from torch.optim import Optimizer

from .tFL import tFL, tFL_Client


class NovaOptimizer(Optimizer):
    """SGD optimizer that accumulates the normalized gradient update for FedNova."""

    def __init__(
        self,
        params: Any,
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        prox_mu: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params=params, defaults=defaults)
        self.prox_mu = prox_mu
        self.momentum = momentum
        self.local_normalizing_vec = 0.0
        self.local_counter = 0.0
        self.local_steps = 0

    def step(self, closure: Any = None) -> Any:
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
    """FedNova: Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization."""

    optional = {
        "gmf": 0.0,
        "prox_mu": 0.0,
        "nova_momentum": 0.0,
    }

    @classmethod
    def args_update(cls, parser: Any) -> None:
        parser.add_argument("--gmf", type=float, default=None)
        parser.add_argument("--prox_mu", type=float, default=None)
        parser.add_argument("--nova_momentum", type=float, default=None)

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        self.parallel = False
        self._global_momentum_buffer: List[torch.Tensor] = []

    def aggregate_client_updates(self, packages: "OrderedDict[int, dict]") -> None:
        total_score = 0.0
        weighted_tau = 0.0
        weighted_gradients: List[torch.Tensor] = []
        for package in packages.values():
            score = float(package["score"])
            total_score += score
            weighted_tau += float(package["tau"]) * score
            if not weighted_gradients:
                weighted_gradients = [
                    gradient * score for gradient in package["nova_grad"]
                ]
            else:
                for accumulated, gradient in zip(
                    weighted_gradients, package["nova_grad"]
                ):
                    accumulated.add_(gradient, alpha=score)
        if total_score <= 0:
            raise ValueError("FedNova client scores must sum to a positive value")

        # τ_eff = weighted_avg(tau_i); d = weighted_avg(cum_grad_i / a_i).
        tau_eff = weighted_tau / total_score
        avg_d = [gradient / total_score for gradient in weighted_gradients]

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
        self._commit_global(new_params=new_params)


class FedNova_Client(tFL_Client):
    """Client for FedNova."""

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
        # The proximal variant retains a_i for d_i but uses local steps for
        # tau_eff, matching the paper and reference optimizer.
        self._tau = (
            nova_opt.local_steps
            if self.prox_mu != 0
            else nova_opt.local_normalizing_vec
        )
        self.model.to("cpu")

    def package(self) -> Dict[str, Any]:
        result = super().package()
        result["regular_model_params"] = {}
        result["nova_grad"] = self._nova_grad
        result["tau"] = self._tau
        result["__wire__"] = ("nova_grad", "tau", "score")
        return result
