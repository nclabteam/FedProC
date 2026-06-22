# -*- coding: utf-8 -*-
"""spFL — Sparse/Pruning Federated Learning base class.

All pruning FL strategies (PruneFL, FedDST, FedTiny, FedMef, FedSGC, FedRTS)
inherit from spFL (server) and spFL_Client (client).

Server manages a global binary mask dict; clients apply it during local training.
Adjustment rounds (every delta_T rounds, up to T_end) trigger mask updates.
"""
from typing import Any, Dict

import torch

from ._spFL_utils import apply_mask, init_mask
from .tFL import tFL, tFL_Client


class spFL(tFL):
    """Sparse FL server base. Manages global mask and delegates mask updates to subclasses."""

    optional = {
        "target_density": 0.5,
        "delta_T": 50,
        "T_end": 500,
        "adjust_alpha": 0.3,
        "pruning_strategy": "ERK_magnitude",
    }

    def __init__(self, configs, times):
        super().__init__(configs, times)
        self._sp_mask_dict: Dict[str, torch.Tensor] = {}
        self._sp_layer_density: Dict[str, float] = {}

    def _sp_is_adj(self) -> bool:
        return (
            self.current_iter > 0
            and self.current_iter % self.delta_T == 0
            and self.current_iter <= self.T_end
        )

    def _sp_init_mask(self) -> None:
        self._sp_mask_dict, self._sp_layer_density = init_mask(
            self.model, self.target_density, self.pruning_strategy
        )
        apply_mask(self.model, self._sp_mask_dict)

    def package(self, client_id: int) -> Dict[str, Any]:
        if not self._sp_mask_dict:
            self._sp_init_mask()
        pkg = super().package(client_id)
        pkg["_sp_mask_dict"] = self._sp_mask_dict
        pkg["_sp_is_adj"] = self._sp_is_adj()
        pkg["_sp_t"] = self.current_iter
        pkg["_sp_T_end"] = self.T_end
        pkg["_sp_alpha"] = self.adjust_alpha
        return pkg

    def aggregate_client_updates(self, packages: Dict[int, Any]) -> None:
        super().aggregate_client_updates(packages)
        apply_mask(self.model, self._sp_mask_dict)
        if self._sp_is_adj():
            self._sp_update_mask(packages)

    def _sp_update_mask(self, packages: Dict[int, Any]) -> None:
        """Override in subclasses to implement strategy-specific mask update."""


class spFL_Client(tFL_Client):
    """Sparse FL client base. Applies mask during local training."""

    def set_parameters(self, package: Dict[str, Any]) -> None:
        super().set_parameters(package)
        self._sp_mask_dict = package["_sp_mask_dict"]
        self._sp_is_adj = package["_sp_is_adj"]
        self._sp_t = package["_sp_t"]
        self._sp_T_end = package["_sp_T_end"]
        self._sp_alpha = package["_sp_alpha"]

    def fit(self) -> None:
        from .base import SharedMethods
        SharedMethods._set_worker_seed(self._loader_seed("train"))
        loader = self.load_train_data()
        offload = self.efficiency == "low"
        apply_mask(self.model, self._sp_mask_dict)
        for _ in range(self.epochs):
            self.train_one_epoch(
                model=self.model,
                dataloader=loader,
                optimizer=self.optimizer,
                criterion=self.loss,
                scheduler=self.scheduler,
                device=self.device,
                offload_after=offload,
            )
            apply_mask(self.model, self._sp_mask_dict)
        if self.efficiency == "med":
            self.model.to("cpu")

    def _collect_gradients(self) -> Dict[str, torch.Tensor]:
        """Single-batch backward pass for gradient growth signal."""
        self.model.to(self.device)
        self.model.train()
        self.model.zero_grad()
        for batch_x, batch_y, x_mark, y_mark in self.load_train_data():
            batch_x = batch_x.to(self.device, dtype=torch.float32)
            batch_y = batch_y.to(self.device, dtype=torch.float32)
            x_mark = x_mark.to(self.device)
            y_mark = y_mark.to(self.device)
            pred = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
            loss = self.loss(pred, batch_y)
            loss.backward()
            break
        grads = {
            n: p.grad.detach().cpu().clone() if p.grad is not None else torch.zeros(p.shape)
            for n, p in self.model.named_parameters()
        }
        self.model.zero_grad()
        return grads
