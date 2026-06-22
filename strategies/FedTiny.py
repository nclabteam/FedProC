# -*- coding: utf-8 -*-
"""FedTiny — Distributed Pruning Towards Tiny Neural Networks in Federated Learning.

Paper: https://arxiv.org/abs/2212.01977  |  ICDCS '23

Client sends post-training gradients (single-batch) on adjustment rounds.
Server: FedAvg of gradients → server-side sparse_update_step (prune mag + grow grad).
"""
from typing import Any, Dict

import torch

from ._spFL_utils import apply_mask, sparse_update_step
from .spFL import spFL, spFL_Client


class FedTiny(spFL):
    """FedTiny server — server-side mask update using federated gradient signal."""

    def _sp_update_mask(self, packages: Dict[int, Any]) -> None:
        total = sum(p["train_samples"] for p in packages.values())
        avg_grad: Dict[str, torch.Tensor] = {}
        for pkg in packages.values():
            w = pkg["train_samples"] / total
            for name, g in pkg["_sp_extra"].items():
                if name not in avg_grad:
                    avg_grad[name] = g.float() * w
                else:
                    avg_grad[name] += g.float() * w

        self._sp_mask_dict = sparse_update_step(
            self.model, avg_grad,
            self._sp_mask_dict,
            self._sp_t, self.T_end, self.adjust_alpha,
        )
        apply_mask(self.model, self._sp_mask_dict)


class FedTiny_Client(spFL_Client):
    """FedTiny client — collects single-batch gradients after training on adj rounds."""

    def package(self, train_time: float) -> Dict[str, Any]:
        result = super().package(train_time)
        result["_sp_extra"] = self._collect_gradients() if self._sp_is_adj else {}
        return result
