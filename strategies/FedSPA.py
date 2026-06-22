# -*- coding: utf-8 -*-
"""FedSPA - Federated Learning with Sparsification-Amplified Privacy and Adaptive Optimization.

Paper: https://arxiv.org/abs/2008.01558  |  IJCAI 2021

Client: at each round, sample a random sparse mask (rand_k, k = p*d active coords,
  same mask across all local iterations); apply scaled noisy SGD:
    theta -= lr * (1/p) * mask * (grad + N(0, sigma^2))
Server: Adam-like adaptive aggregation of model updates (first + second moments
  of avg delta, not of delta^2 directly):
    u_t = beta1*u_{t-1} + (1-beta1)*avg_delta
    v_t = beta2*v_{t-1} + (1-beta2)*u_t^2
    theta_{t+1} = theta_t + global_lr * u_t / (sqrt(v_t) + kappa)
"""
from collections import OrderedDict
from typing import Any, Dict

import torch

from .tFL import tFL, tFL_Client


class FedSPA(tFL):
    """FedSPA server - Adam-like adaptive aggregation over sparsified DP client updates."""

    optional = {
        "dp_sigma": 0.1,
        "compression_ratio": 0.1,
        "global_lr": 1e-3,
        "beta1": 0.9,
        "beta2": 0.999,
        "kappa": 1e-3,
    }

    def __init__(self, configs, times):
        super().__init__(configs, times)
        self._spa_u: Dict[str, torch.Tensor] = {}
        self._spa_v: Dict[str, torch.Tensor] = {}

    def _spa_init_moments(self) -> None:
        for n, p in self.model.named_parameters():
            self._spa_u[n] = torch.zeros_like(p.data)
            self._spa_v[n] = torch.full_like(p.data, self.kappa ** 2)

    def package(self, client_id: int) -> Dict[str, Any]:
        pkg = super().package(client_id)
        pkg["_spa_sigma"] = self.dp_sigma
        pkg["_spa_p"] = self.compression_ratio
        return pkg

    def aggregate_client_updates(self, packages: "OrderedDict[int, dict]") -> None:
        if not self._spa_u:
            self._spa_init_moments()

        scores = [p["score"] for p in packages.values()]
        total = float(sum(scores))

        avg_delta: Dict[str, torch.Tensor] = {}
        for pkg, score in zip(packages.values(), scores):
            w = score / total
            for n, client_param in pkg["regular_model_params"].items():
                delta = client_param.float() - self.public_model_params[n].float()
                if n not in avg_delta:
                    avg_delta[n] = delta * w
                else:
                    avg_delta[n] += delta * w

        new_params = OrderedDict()
        for n, param in self.model.named_parameters():
            if n not in avg_delta:
                new_params[n] = self.public_model_params[n]
                continue
            d = avg_delta[n].to(param.device)
            self._spa_u[n] = self.beta1 * self._spa_u[n] + (1 - self.beta1) * d
            self._spa_v[n] = self.beta2 * self._spa_v[n] + (1 - self.beta2) * self._spa_u[n] ** 2
            new_params[n] = (
                self.public_model_params[n].to(param.device)
                + self.global_lr * self._spa_u[n] / (self._spa_v[n].sqrt() + self.kappa)
            ).cpu()

        self._commit_global(new_params)


class FedSPA_Client(tFL_Client):
    """FedSPA client - random sparsification + Gaussian DP noise during local SGD."""

    def set_parameters(self, package: Dict[str, Any]) -> None:
        super().set_parameters(package)
        self._spa_sigma = package["_spa_sigma"]
        self._spa_p = package["_spa_p"]

    def fit(self) -> None:
        from .base import SharedMethods
        SharedMethods._set_worker_seed(self._loader_seed("train"))

        # Build per-round random sparse mask (same across all local iterations)
        named_params = list(self.model.named_parameters())
        d = sum(p.numel() for _, p in named_params)
        k = max(1, int(self._spa_p * d))
        active = torch.randperm(d)[:k]
        active_set = set(active.tolist())

        sparse_mask: Dict[str, torch.Tensor] = {}
        offset = 0
        for n, p in named_params:
            n_elem = p.numel()
            local_active = [i - offset for i in active_set if offset <= i < offset + n_elem]
            mask = torch.zeros(n_elem, dtype=torch.float32)
            if local_active:
                mask[torch.tensor(local_active)] = 1.0
            sparse_mask[n] = mask.view(p.shape)
            offset += n_elem

        self.model.to(self.device)
        self.model.train()
        loader = self.load_train_data()
        lr = self.optimizer.param_groups[0]["lr"]
        scale = 1.0 / self._spa_p if self._spa_p > 0 else 1.0
        offload = self.efficiency == "low"

        for _ in range(self.epochs):
            for batch_x, batch_y, x_mark, y_mark in loader:
                batch_x = batch_x.to(self.device, dtype=torch.float32)
                batch_y = batch_y.to(self.device, dtype=torch.float32)
                x_mark = x_mark.to(self.device)
                y_mark = y_mark.to(self.device)

                self.optimizer.zero_grad()
                pred = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
                loss = self.loss(pred, batch_y)
                loss.backward()

                with torch.no_grad():
                    for n, p in self.model.named_parameters():
                        if p.grad is None:
                            continue
                        mask = sparse_mask[n].to(p.device)
                        noise = torch.randn_like(p.grad) * self._spa_sigma
                        p.data -= lr * scale * mask * (p.grad + noise)

                if self.scheduler:
                    self.scheduler.step()

            if offload:
                self.model.to("cpu")

        if self.efficiency == "med":
            self.model.to("cpu")
