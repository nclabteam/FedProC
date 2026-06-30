# -*- coding: utf-8 -*-
"""FedFew: Few-for-Many Personalized Federated Learning.

Maintains K server models. Each client trains all K models locally and
sends per-model gradients + losses. Server aggregates via Smooth
Tchebycheff Set (STCH-Set) scalarization.

Weights (paper §3.2, Eqs. 4-5):
  S_i      = Σ_k exp(-n̄_i L_i(θ_k) / μ)          [n̄_i = normalized sample size]
  α_i      = S_i^{-1} / Σ_j S_j^{-1}               [outer: hard-sample mining]
  w_{ik}   = exp(-n̄_i L_i(θ_k) / μ) / S_i          [inner: soft model selection]
  θ_k^t   = θ_k^{t-1} - lr · Σ_i α_i w_{ik} g_{ik} [server gradient step]

Paper: arxiv.org/abs/2603.11992  (lib ID 08832)
"""
import copy
from collections import OrderedDict
from typing import Any, Dict, List

import torch

from .pFL import pFL, pFL_Client


class FedFew(pFL):
    """FedFew server."""

    optional = {"num_models": 3, "mu": 1.0}

    def __init__(self, configs, times) -> None:
        super().__init__(configs, times)
        self.server_models: List[OrderedDict] = [
            OrderedDict({k: v.clone() for k, v in self.public_model_params.items()})
            for _ in range(self.num_models)
        ]

    def package(self, client_id: int) -> dict:
        pkg = super().package(client_id)
        pkg["fedfew_server_models"] = self.server_models
        return pkg

    def aggregate_client_updates(self, packages) -> None:
        K = self.num_models
        cids = list(packages.keys())

        # Normalized sample sizes n̄_i = n_i / Σ_j n_j  (paper footnote §3.2)
        total_samples = sum(packages[cid]["score"] for cid in cids)

        # S_i = Σ_k exp(-n̄_i · L_i(θ_k) / μ)
        S: Dict[int, float] = {}
        for cid in cids:
            n_bar = packages[cid]["score"] / total_samples
            losses = torch.tensor(packages[cid]["fedfew_losses"], dtype=torch.float)
            S[cid] = torch.exp(-n_bar * losses / self.mu).sum().item()

        # α_i = S_i^{-1} / Σ_j S_j^{-1}
        sum_S_inv = sum(1.0 / max(S[cid], 1e-8) for cid in cids)
        alpha: Dict[int, float] = {cid: (1.0 / max(S[cid], 1e-8)) / sum_S_inv for cid in cids}

        # w_{ik} = exp(-n̄_i L_i(θ_k) / μ) / S_i
        w: Dict[int, List[float]] = {}
        for cid in cids:
            n_bar = packages[cid]["score"] / total_samples
            losses = torch.tensor(packages[cid]["fedfew_losses"], dtype=torch.float)
            w[cid] = (torch.exp(-n_bar * losses / self.mu) / max(S[cid], 1e-8)).tolist()

        # θ_k^t = θ_k^{t-1} - lr · Σ_i α_i w_{ik} g_{ik}
        for k in range(K):
            for name in self.server_models[k]:
                grad_agg = torch.zeros_like(self.server_models[k][name], dtype=torch.float)
                for cid in cids:
                    grad_agg.add_(
                        packages[cid]["fedfew_gradients"][k][name].float(),
                        alpha=alpha[cid] * w[cid][k],
                    )
                self.server_models[k][name] = (
                    self.server_models[k][name].float() - self.learning_rate * grad_agg
                )

        # Personal model: best-matching model for each participating client
        for cid in cids:
            best_k = int(torch.tensor(packages[cid]["fedfew_losses"]).argmin().item())
            self.clients_personal_model_params[cid].update(
                {k: v.clone() for k, v in self.server_models[best_k].items()}
            )

        # Global model: average of K server models (used for generalization eval)
        new_global = OrderedDict(
            (name, torch.stack([self.server_models[k][name].float() for k in range(K)]).mean(0))
            for name in self.public_model_params
        )
        self._commit_global(new_global)


class FedFew_Client(pFL_Client):
    """FedFew client — trains all K server models locally."""

    _server_models: List[OrderedDict] = None
    _fedfew_losses: List[float] = None
    _fedfew_gradients: List[Dict[str, torch.Tensor]] = None

    def set_parameters(self, package: dict) -> None:
        super().set_parameters(package)
        self._server_models = package["fedfew_server_models"]

    def fit(self) -> None:
        self._fedfew_losses = []
        self._fedfew_gradients = []

        for model_params in self._server_models:
            model_k = copy.deepcopy(self.model)
            model_k.load_state_dict(model_params, strict=False)
            optimizer_k = type(self.optimizer)(
                model_k.parameters(), lr=self.learning_rate
            )

            loader = self.load_train_data()
            for _ in range(self.local_epochs):
                self.train_one_epoch(
                    model=model_k,
                    dataloader=loader,
                    optimizer=optimizer_k,
                    criterion=self.loss,
                    scheduler=None,
                    device=self.device,
                )

            # g_{ik} = ∇L_i(θ_k) at final local state (paper Alg. 1 line 7)
            # Accumulate gradients over full training loader then average
            model_k.zero_grad()
            n_batches = 0
            total_loss = 0.0
            for batch in self.load_train_data():
                batch_x = batch[0].to(self.device)
                batch_y = batch[1].to(self.device)
                pred = model_k(batch_x)
                loss = self.loss(pred, batch_y)
                loss.backward()
                total_loss += loss.item()
                n_batches += 1

            scale = 1.0 / max(n_batches, 1)
            self._fedfew_losses.append(total_loss * scale)
            self._fedfew_gradients.append({
                n: (p.grad.detach().cpu() * scale if p.grad is not None
                    else torch.zeros_like(p.data).cpu())
                for n, p in model_k.named_parameters()
            })

        # Load best model into self.model so standard package() captures it
        best_k = int(torch.tensor(self._fedfew_losses).argmin().item())
        self.model.load_state_dict(self._server_models[best_k], strict=False)

    def package(self) -> Dict[str, Any]:
        result = super().package()
        result["fedfew_losses"] = self._fedfew_losses
        result["fedfew_gradients"] = self._fedfew_gradients
        return result
