# -*- coding: utf-8 -*-
"""PruneFL — Model Pruning Enables Efficient Federated Learning on Edge Devices.

Paper: https://arxiv.org/abs/1909.12326  |  TNNLS '22

Server-side mask update: FedAvg of per-client squared gradients (grad²) →
    prune k smallest-magnitude active weights + grow k largest-grad² inactive weights.
k decays via cosine schedule (f_decay).
"""
from typing import Any, Dict

import torch

from .spFL import spFL, spFL_Client


class PruneFL(spFL):
    """PruneFL server — uses federated squared gradients to guide mask updates."""

    def _sp_update_mask(self, packages: Dict[int, Any]) -> None:
        total = sum(p["train_samples"] for p in packages.values())
        avg_grad_sq: Dict[str, torch.Tensor] = {}
        for pkg in packages.values():
            w = pkg["train_samples"] / total
            for name, gsq in pkg["_sp_extra"].items():
                if name not in avg_grad_sq:
                    avg_grad_sq[name] = gsq.float() * w
                else:
                    avg_grad_sq[name] += gsq.float() * w

        self._sp_mask_dict = self.sparse_update_step(
            self.model, avg_grad_sq,
            self._sp_mask_dict,
            self._sp_t, self.T_end, self.adjust_alpha,
        )
        self.apply_mask(self.model, self._sp_mask_dict)


class PruneFL_Client(spFL_Client):
    """PruneFL client — accumulates squared gradients during training on adj rounds."""

    def fit(self) -> None:
        from .base import SharedMethods
        SharedMethods._set_worker_seed(self._loader_seed("train"))
        loader = self.load_train_data()
        offload = self.efficiency == "low"
        self.apply_mask(self.model, self._sp_mask_dict)

        grad_sq: Dict[str, torch.Tensor] = {}
        n_steps = 0

        for _ in range(self.epochs):
            self.model.to(self.device)
            self.model.train()
            for batch_x, batch_y, x_mark, y_mark in loader:
                batch_x = batch_x.to(self.device, dtype=torch.float32)
                batch_y = batch_y.to(self.device, dtype=torch.float32)
                x_mark = x_mark.to(self.device)
                y_mark = y_mark.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
                loss = self.loss(pred, batch_y)
                loss.backward()
                if self._sp_is_adj:
                    for n, p in self.model.named_parameters():
                        g = p.grad
                        if g is not None:
                            gsq = g.detach().cpu().pow(2)
                            if n not in grad_sq:
                                grad_sq[n] = gsq
                            else:
                                grad_sq[n] += gsq
                    n_steps += 1
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
            self.apply_mask(self.model, self._sp_mask_dict)
            if offload:
                self.model.to("cpu")

        if self.efficiency == "med":
            self.model.to("cpu")

        if self._sp_is_adj and n_steps > 0:
            self._sp_grad_sq = {n: v / n_steps for n, v in grad_sq.items()}
        else:
            self._sp_grad_sq = {}

    def package(self, train_time: float) -> Dict[str, Any]:
        result = super().package(train_time)
        result["_sp_extra"] = self._sp_grad_sq if self._sp_is_adj else {}
        return result
