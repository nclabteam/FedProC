# -*- coding: utf-8 -*-
"""FedSPA - Federated Learning with Sparsification-Amplified Privacy and Adaptive Optimization."""

from collections import OrderedDict
from typing import Any, Dict

import torch

from .tFL import tFL, tFL_Client


class FedSPA(tFL):
    """FedSPA server - Adam-like adaptive aggregation over sparsified DP client updates."""

    optional = {
        "dp_sigma": 0.1,
        "compression_ratio": 0.1,
        "global_lr": 1e-2,
        "beta1": 0.9,
        "beta2": 0.99,
        "kappa": 1e-3,
    }

    def __init__(self, configs: Any, times: Any) -> None:
        super().__init__(configs=configs, times=times)
        self._spa_u: Dict[str, torch.Tensor] = {}
        self._spa_v: Dict[str, torch.Tensor] = {}

    def _spa_init_moments(self) -> None:
        for n, p in self.model.named_parameters():
            self._spa_u[n] = torch.zeros_like(p.data)
            self._spa_v[n] = torch.full_like(p.data, self.kappa**2)

    def package(self, client_id: int) -> Dict[str, Any]:
        pkg = super().package(client_id=client_id)
        pkg["_spa_sigma"] = self.dp_sigma
        pkg["_spa_p"] = self.compression_ratio
        return pkg

    def aggregate_client_updates(self, packages: "OrderedDict[int, dict]") -> None:
        if not self._spa_u:
            self._spa_init_moments()

        avg_delta = {
            n: torch.zeros_like(param, dtype=torch.float32)
            for n, param in self.public_model_params.items()
        }
        for pkg in packages.values():
            for n, (indices, values) in pkg["spa_delta"].items():
                flat = avg_delta[n].view(-1)
                flat.index_add_(
                    0,
                    indices.to(device=flat.device, dtype=torch.long),
                    values.to(device=flat.device, dtype=flat.dtype),
                )
        for delta in avg_delta.values():
            delta.div_(len(packages))

        new_params = OrderedDict()
        for n, param in self.model.named_parameters():
            if n not in avg_delta:
                new_params[n] = self.public_model_params[n]
                continue
            d = avg_delta[n].to(param.device)
            self._spa_u[n] = self.beta1 * self._spa_u[n] + (1 - self.beta1) * d
            self._spa_v[n] = (
                self.beta2 * self._spa_v[n] + (1 - self.beta2) * self._spa_u[n] ** 2
            )
            new_params[n] = (
                self.public_model_params[n].to(param.device)
                + self.global_lr * self._spa_u[n] / (self._spa_v[n].sqrt() + self.kappa)
            ).cpu()

        self._commit_global(new_params=new_params)


class FedSPA_Client(tFL_Client):
    """FedSPA client - random sparsification + Gaussian DP noise during local SGD."""

    def set_parameters(self, package: Dict[str, Any]) -> None:
        super().set_parameters(package=package)
        self._spa_sigma = package["_spa_sigma"]
        self._spa_p = package["_spa_p"]
        if not 0 < self._spa_p <= 1:
            raise ValueError("FedSPA compression_ratio must be in (0, 1]")
        self._spa_initial_params = {
            n: p.detach().cpu().clone() for n, p in self.model.named_parameters()
        }

    def fit(self) -> None:
        self._set_worker_seed(seed=self._loader_seed(dataset_type="train"))

        # Build per-round random sparse mask (same across all local iterations)
        named_params = list(self.model.named_parameters())
        d = sum(p.numel() for _, p in named_params)
        k = max(1, int(self._spa_p * d))
        active = torch.randperm(d)[:k]
        flat_mask = torch.zeros(d, dtype=torch.bool)
        flat_mask[active] = True

        sparse_mask: Dict[str, torch.Tensor] = {}
        offset = 0
        for n, p in named_params:
            n_elem = p.numel()
            sparse_mask[n] = flat_mask[offset : offset + n_elem].view(p.shape)
            offset += n_elem
        self._spa_mask = sparse_mask

        self.model.to(self.device)
        self.model.train()
        device_masks = {n: mask.to(self.device) for n, mask in sparse_mask.items()}
        loader = self.load_train_data()
        self.initialize_scheduler(steps_per_epoch=len(loader))
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
                    learning_rate = self.optimizer.param_groups[0]["lr"]
                    for n, p in self.model.named_parameters():
                        if p.grad is None:
                            continue
                        active = device_masks[n]
                        noisy_grad = p.grad[active] + (
                            torch.randn_like(p.grad[active]) * self._spa_sigma
                        )
                        p.data[active] = (
                            p.data[active] - learning_rate * scale * noisy_grad
                        )

                self.step_scheduler_batch(
                    scheduler=self.scheduler,
                    batch_data=batch_x,
                )

            self.step_scheduler_epoch(scheduler=self.scheduler)
            if offload:
                self.model.to("cpu")

        if self.efficiency == "med":
            self.model.to("cpu")

    def package(self) -> Dict[str, Any]:
        result = super().package()
        sparse_delta = {}
        for n, current in result["regular_model_params"].items():
            mask = self._spa_mask[n].reshape(-1)
            initial = self._spa_initial_params[n].float().reshape(-1)
            delta = current.float().reshape(-1) - initial
            indices = mask.nonzero(as_tuple=False).reshape(-1).to(torch.int32)
            sparse_delta[n] = (indices, delta[indices])
        result["regular_model_params"] = {}
        result["spa_delta"] = sparse_delta
        result["__wire__"] = ("spa_delta",)
        return result
