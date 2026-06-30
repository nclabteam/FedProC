import copy
from argparse import Namespace
from collections import OrderedDict
from typing import Any, Dict

import torch

from .tFL import tFL, tFL_Client


class FedADMM(tFL):
    """FedADMM: Federated Learning via ADMM (Elgabli et al., ICDE 2022).

    Solves the consensus problem: minimize Σ f_i(w_i) s.t. w_i = θ via ADMM
    in scaled-dual form with per-client variable α_i = -u_i (scaled dual):

      Client proximal loss: f_i(w) + (ρ/2)||w - θ - α_i||²
      Dual update:          α_i ← α_i - (w_i - θ)
      Theta update:         θ  ← weighted_mean(w_i - α_i)
      Global model:         weighted_mean(w_i)    [for eval/broadcast]

    Paper differences:
    - θ-update: paper uses a tracking rule (θ ← θ + Σ Δ_i); here we use the
      standard ADMM consensus update (mean(w_i - α_i)), simpler and valid for
      partial participation.
    - Per-client w_i: clients maintain their own persistent local model across
      rounds (paper sends per-client w_i, not the global average), which is
      necessary for the dual variables to work correctly.

    Default hyperparameters (from paper): ρ = 0.01.
    Reference: arXiv:2204.03529. ICDE 2022.
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
        zeros = {
            name: torch.zeros_like(p.data).cpu()
            for name, p in self.model.named_parameters()
        }
        init_w = {k: v.clone() for k, v in self.public_model_params.items()}
        for cid in range(self.num_clients):
            self.clients_personal_model_params[cid]["alpha"] = {
                name: v.clone() for name, v in zeros.items()
            }
            # Each client keeps its own local model w_i across rounds (paper Alg. 1)
            self.clients_personal_model_params[cid]["w_i"] = {
                k: v.clone() for k, v in init_w.items()
            }

    def package(self, client_id: int) -> Dict[str, Any]:
        out = super().package(client_id)
        out["theta"] = copy.deepcopy(self.theta)
        # Client starts from its own persistent w_i (not the global average)
        out["regular_model_params"] = copy.deepcopy(
            self.clients_personal_model_params[client_id]["w_i"]
        )
        return out

    def aggregate_client_updates(self, packages: "OrderedDict[int, dict]") -> None:
        scores = [p["score"] for p in packages.values()]
        total = float(sum(scores))
        weights = [s / total for s in scores]

        # Persist each client's updated w_i for the next round (paper Alg. 1, line 8)
        for cid, pkg in packages.items():
            self.clients_personal_model_params[cid]["w_i"] = {
                k: v.clone() for k, v in pkg["regular_model_params"].items()
            }

        # Global model = weighted avg of w_i (used for evaluation and θ init)
        new_global = OrderedDict()
        for name in self.public_model_params:
            stacked = torch.stack(
                [p["regular_model_params"][name] for p in packages.values()], dim=-1
            )
            w_t = torch.tensor(weights, dtype=stacked.dtype)
            new_global[name] = (stacked * w_t).sum(dim=-1)

        # θ update: weighted mean(w_i - α_i)  [consensus ADMM, not paper's tracking rule]
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
    """Client for FedADMM.

    Maintains per-parameter scaled dual variable α_i (server initializes to zeros,
    stored in personal_model_params["alpha"]). Each round, trains with the ADMM
    proximal loss and returns the updated α.
    """

    def set_parameters(self, package: Dict[str, Any]) -> None:
        super().set_parameters(package)
        self._alpha: Dict[str, torch.Tensor] = {
            name: t.clone() for name, t in package["personal_model_params"]["alpha"].items()
        }
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

                # ADMM proximal term: (ρ/2)||w - θ - α||²
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

    def package(self) -> Dict[str, Any]:
        out = super().package()

        # Dual update: α_i ← α_i - (w_i - θ)  [scaled-form ADMM consensus]
        updated_alpha: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                w_i = param.data.cpu()
                updated_alpha[name] = self._alpha[name] - w_i + self._theta[name]

        out["personal_model_params"]["alpha"] = updated_alpha
        return out
