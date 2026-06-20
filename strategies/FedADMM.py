import copy
from argparse import Namespace
from collections import OrderedDict
from typing import Any, Dict

import torch

from .tFL import tFL, tFL_Client


class FedADMM(tFL):
    """
    FedADMM: Federated Learning via Alternating Direction Method of Multipliers.

    Server maintains theta (consensus variable, one tensor per model parameter).
    Each round:
      1. Server sends global model + theta to each client.
      2. Client trains with ADMM proximal loss:
           L(w) + (rho/2) * ||w - theta - alpha||^2
      3. Client updates dual variable: alpha += w - theta.
      4. Server aggregates:
           theta_new[p] = weighted_mean(w_i[p] - alpha_i[p])
           global[p]    = weighted_mean(w_i[p])

    Reference: fl-bench FedADMM implementation.
    """

    optional = {
        "rho": 0.01,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--rho", type=float, default=None)

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        self.parallel = False
        self.theta: Dict[str, torch.Tensor] = {
            name: p.data.clone().cpu()
            for name, p in self.model.named_parameters()
        }

    def package(self, client_id: int) -> Dict[str, Any]:
        out = super().package(client_id)
        out["theta"] = copy.deepcopy(self.theta)
        return out

    def aggregate_client_updates(self, packages: "OrderedDict[int, dict]") -> None:
        scores = [p["score"] for p in packages.values()]
        total = float(sum(scores))
        weights = [s / total for s in scores]

        # New global params: weighted mean of client model weights
        new_global = OrderedDict()
        for name in self.public_model_params:
            stacked = torch.stack(
                [p["regular_model_params"][name] for p in packages.values()], dim=-1
            )
            w_t = torch.tensor(weights, dtype=stacked.dtype)
            new_global[name] = (stacked * w_t).sum(dim=-1)

        # Update theta: theta[p] = weighted_mean(w_i[p] - alpha_i[p])
        new_theta: Dict[str, torch.Tensor] = {}
        for name in self.theta:
            diffs = []
            for pkg in packages.values():
                w_i = pkg["regular_model_params"][name]
                alpha_i = pkg["personal_model_params"]["alpha"][name]
                diffs.append(w_i - alpha_i)
            stacked = torch.stack(diffs, dim=-1)
            w_t = torch.tensor(weights, dtype=stacked.dtype)
            new_theta[name] = (stacked * w_t).sum(dim=-1)
        self.theta = new_theta

        self._commit_global(new_global)


class FedADMM_Client(tFL_Client):
    """
    Client for FedADMM.

    Maintains per-parameter dual variable alpha (stored in
    clients_personal_model_params under key "alpha").  Each round, trains with
    the ADMM proximal loss and returns the updated alpha.
    """

    def set_parameters(self, package: Dict[str, Any]) -> None:
        super().set_parameters(package)
        alpha_data = package["personal_model_params"].get("alpha", None)
        if alpha_data is None:
            self._alpha: Dict[str, torch.Tensor] = {
                name: torch.zeros_like(p.data).cpu()
                for name, p in self.model.named_parameters()
            }
        else:
            self._alpha = {name: t.clone() for name, t in alpha_data.items()}
        self._theta: Dict[str, torch.Tensor] = {
            name: t.clone() for name, t in package["theta"].items()
        }

    def fit(self) -> None:
        self._set_worker_seed(self._loader_seed("train"))
        loader = self.load_train_data()

        self.model.to(self.device)
        self.model.train()

        for _ in range(self.epochs):
            for batch_x, batch_y, x_mark, y_mark in loader:
                self.optimizer.zero_grad()
                batch_x = batch_x.to(device=self.device, dtype=torch.float32)
                batch_y = batch_y.to(device=self.device, dtype=torch.float32)
                x_mark = x_mark.to(device=self.device, dtype=torch.float32)
                y_mark = y_mark.to(device=self.device, dtype=torch.float32)

                outputs = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
                loss = self.loss(outputs, batch_y)

                # ADMM proximal term: (rho/2) * ||w - theta - alpha||^2
                for name, param in self.model.named_parameters():
                    theta_i = self._theta[name].to(self.device)
                    alpha_i = self._alpha[name].to(self.device)
                    proximal = param - theta_i - alpha_i
                    loss = loss + (self.rho / 2.0) * proximal.pow(2).sum()

                loss.backward()
                self.optimizer.step()

            self.scheduler.step()

        if self.efficiency != "high":
            self.model.to("cpu")

    def package(self, train_time: float) -> Dict[str, Any]:
        out = super().package(train_time)

        # Update alpha: alpha_i += w_i - theta_i
        updated_alpha: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                w_i = param.data.cpu()
                updated_alpha[name] = self._alpha[name] + w_i - self._theta[name]

        out["personal_model_params"]["alpha"] = updated_alpha
        return out
