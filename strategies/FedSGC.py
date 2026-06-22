# -*- coding: utf-8 -*-
"""FedSGC - Gradient-Congruity Guided Federated Sparse Training.

Paper: https://arxiv.org/abs/2405.01189  |  ICLRW '24

Client-local mask update guided by agreement between global and local weight change directions.
Prune weights where global/local directions conflict; grow where they agree.
Server: OR-union of client masks -> magnitude re-prune; maintains global direction map for next round.
"""
from typing import Any, Dict, Optional

import torch

from .spFL import spFL, spFL_Client


class FedSGC(spFL):
    """FedSGC server - OR-union mask + global direction map for client guidance."""

    optional = {
        **spFL.optional,
    }

    def __init__(self, configs, times):
        super().__init__(configs, times)
        self._prev_global_params: Optional[Dict[str, torch.Tensor]] = None
        self._global_direction_map: Dict[str, torch.Tensor] = {}

    def package(self, client_id: int) -> Dict[str, Any]:
        pkg = super().package(client_id)
        pkg["_sp_global_direction_map"] = self._global_direction_map
        return pkg

    def _sp_update_mask(self, packages: Dict[int, Any]) -> None:
        client_masks = [pkg["_sp_extra"]["mask_dict"] for pkg in packages.values()
                        if "_sp_extra" in pkg and "mask_dict" in pkg["_sp_extra"]]
        if not client_masks:
            return
        unioned = self.union_masks(client_masks)
        self._sp_mask_dict = self.magnitude_reprune(self.model, unioned, self._sp_layer_density)
        self.apply_mask(self.model, self._sp_mask_dict)

    def aggregate_client_updates(self, packages: Dict[int, Any]) -> None:
        cur_params = {n: p.data.cpu().clone() for n, p in self.model.named_parameters()}
        super().aggregate_client_updates(packages)
        new_params = {n: p.data.cpu().clone() for n, p in self.model.named_parameters()}
        self._global_direction_map = {
            n: torch.sign(new_params[n] - cur_params[n]) for n in new_params
        }


class FedSGC_Client(spFL_Client):
    """FedSGC client - direction-coherent local mask update."""

    optional = {
        "A_epochs": None,
        "lambda_param": 0.5,
        "beta_param": 0.5,
    }

    def set_parameters(self, package: Dict[str, Any]) -> None:
        super().set_parameters(package)
        self._sp_global_dir = package.get("_sp_global_direction_map", {})

    def fit(self) -> None:
        from .base import SharedMethods
        SharedMethods._set_worker_seed(self._loader_seed("train"))
        loader = self.load_train_data()
        offload = self.efficiency == "low"
        self.apply_mask(self.model, self._sp_mask_dict)

        total_epochs = self.epochs
        a_epochs = self.A_epochs if self.A_epochs is not None else total_epochs // 2 + 1

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

        if not self._sp_is_adj:
            _run_epochs(total_epochs)
            if self.efficiency == "med":
                self.model.to("cpu")
            return

        # Snapshot weights before training
        init_w = {n: p.data.cpu().clone() for n, p in self.model.named_parameters()}
        _run_epochs(min(a_epochs, total_epochs))

        # local direction map
        local_dir = {
            n: torch.sign(p.data.cpu() - init_w[n])
            for n, p in self.model.named_parameters()
        }

        # collect gradients for growth signal
        grads = self._collect_gradients()

        # direction-coherent prune + grow
        self._sp_mask_dict = _prune_grow_sgc(
            self.model, self._sp_mask_dict, grads, local_dir, self._sp_global_dir,
            self._sp_t, self._sp_T_end, self._sp_alpha,
            self.lambda_param, self.beta_param,
        )
        self.apply_mask(self.model, self._sp_mask_dict)

        remaining = total_epochs - min(a_epochs, total_epochs)
        if remaining > 0:
            _run_epochs(remaining)

        if self.efficiency == "med":
            self.model.to("cpu")

    def package(self, train_time: float) -> Dict[str, Any]:
        result = super().package(train_time)
        result["_sp_extra"] = (
            {"mask_dict": {n: m.cpu() for n, m in self._sp_mask_dict.items()}}
            if self._sp_is_adj else {}
        )
        return result


def _prune_grow_sgc(
    model: torch.nn.Module,
    mask_dict: Dict[str, torch.Tensor],
    grads: Dict[str, torch.Tensor],
    local_dir: Dict[str, torch.Tensor],
    global_dir: Dict[str, torch.Tensor],
    t: int, T_end: int, alpha: float,
    lambda_k: float, beta_k: float,
) -> Dict[str, torch.Tensor]:
    for name, param in model.named_parameters():
        if name not in mask_dict:
            continue
        mask = mask_dict[name].view(-1)
        w = param.data.abs().view(-1).cpu()
        g = grads.get(name, torch.zeros_like(w)).abs().view(-1)
        ld = local_dir.get(name, torch.zeros_like(w)).view(-1)
        gd = global_dir.get(name, torch.zeros_like(w)).view(-1)

        active_idx = (mask == 1).nonzero(as_tuple=False).view(-1)
        inactive_idx = (mask == 0).nonzero(as_tuple=False).view(-1)
        active_num = len(active_idx)
        k = int(spFL.f_decay(t, alpha, T_end) * active_num)
        if k <= 0:
            continue

        # Prune: conflicting direction first, then remaining
        conflict = active_idx[(gd[active_idx] == -ld[active_idx])]
        non_conflict = active_idx[(gd[active_idx] != -ld[active_idx])]

        n1 = min(int(lambda_k * k), len(conflict))
        if n1 > 0:
            _, rel = torch.topk(w[conflict], n1, largest=False)
            mask[conflict[rel]] = 0.0

        n2 = min(int((1 - lambda_k) * k), len(non_conflict))
        if n2 > 0:
            _, rel = torch.topk(w[non_conflict], n2, largest=False)
            mask[non_conflict[rel]] = 0.0

        # Grow: agreement first, then remaining (update inactive after prune)
        inactive_idx = (mask == 0).nonzero(as_tuple=False).view(-1)
        agree = inactive_idx[(gd[inactive_idx] == ld[inactive_idx])]
        disagree = inactive_idx[(gd[inactive_idx] != ld[inactive_idx])]

        n3 = min(int(beta_k * k), len(agree))
        if n3 > 0:
            _, rel = torch.topk(g[agree], n3, largest=True)
            mask[agree[rel]] = 1.0

        n4 = min(int((1 - beta_k) * k), len(disagree))
        if n4 > 0:
            _, rel = torch.topk(g[disagree], n4, largest=True)
            mask[disagree[rel]] = 1.0

        mask_dict[name] = mask.view(mask_dict[name].shape)

    return mask_dict
