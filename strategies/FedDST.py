# -*- coding: utf-8 -*-
"""FedDST — Federated Dynamic Sparse Training.

Paper: https://arxiv.org/abs/2112.09824  |  AAAI '22

Client-local mask update: train A_epochs → compute local gradient → call
    sparse_update_step locally → continue training remaining epochs.
Server: OR-union of client masks → magnitude re-prune to target density.
"""
from typing import Any, Dict

import torch

from .spFL import spFL, spFL_Client


class FedDST(spFL):
    """FedDST server — reconciles diverged client masks via OR-union + magnitude re-prune."""

    def _sp_update_mask(self, packages: Dict[int, Any]) -> None:
        client_masks = [pkg["_sp_extra"]["mask_dict"] for pkg in packages.values()
                        if "_sp_extra" in pkg and "mask_dict" in pkg["_sp_extra"]]
        if not client_masks:
            return
        unioned = self.union_masks(client_masks)
        self._sp_mask_dict = self.magnitude_reprune(self.model, unioned, self._sp_layer_density)
        self.apply_mask(self.model, self._sp_mask_dict)


class FedDST_Client(spFL_Client):
    """FedDST client — local dynamic sparse training with A_epochs split."""

    optional = {"A_epochs": None}

    def fit(self) -> None:
        from .base import SharedMethods
        SharedMethods._set_worker_seed(self._loader_seed("train"))
        loader = self.load_train_data()
        offload = self.efficiency == "low"
        self.apply_mask(self.model, self._sp_mask_dict)

        total_epochs = self.epochs
        a_epochs = self.A_epochs if self.A_epochs is not None else total_epochs // 2

        def _run_epochs(n):
            for _ in range(n):
                self.train_one_epoch(
                    model=self.model,
                    dataloader=loader,
                    optimizer=self.optimizer,
                    criterion=self.loss,
                    scheduler=self.scheduler,
                    device=self.device,
                    offload_after=offload,
                )
                self.apply_mask(self.model, self._sp_mask_dict)

        if self._sp_is_adj:
            _run_epochs(min(a_epochs, total_epochs))
            grads = self._collect_gradients()
            self._sp_mask_dict = self.sparse_update_step(
                self.model, grads, self._sp_mask_dict,
                self._sp_t, self._sp_T_end, self._sp_alpha,
            )
            self.apply_mask(self.model, self._sp_mask_dict)
            remaining = total_epochs - min(a_epochs, total_epochs)
            if remaining > 0:
                _run_epochs(remaining)
        else:
            _run_epochs(total_epochs)

        if self.efficiency == "med":
            self.model.to("cpu")

    def package(self, train_time: float) -> Dict[str, Any]:
        result = super().package(train_time)
        result["_sp_extra"] = (
            {"mask_dict": {n: m.cpu() for n, m in self._sp_mask_dict.items()}}
            if self._sp_is_adj else {}
        )
        return result
