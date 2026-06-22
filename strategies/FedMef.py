# -*- coding: utf-8 -*-
"""FedMef — Towards Memory-efficient Federated Dynamic Pruning.

Paper: https://arxiv.org/abs/2403.14737  |  CVPR '24

Extends FedTiny with BAE (Bias-Aware Estimation) regularization:
    L_total = L_task + λ ||w_{low-mag active}||²
Server optionally filters client gradients to top-k inactive (enable_topk_grad).
"""
from typing import Any, Dict, Optional

import torch

from ._spFL_utils import apply_mask, f_decay, sparse_update_step
from .spFL import spFL, spFL_Client


class FedMef(spFL):
    """FedMef server — FedTiny mask update with optional topk gradient filtering."""

    optional = {
        **spFL.optional,
        "enable_topk_grad": False,
    }

    def _sp_update_mask(self, packages: Dict[int, Any]) -> None:
        total = sum(p["train_samples"] for p in packages.values())
        avg_grad: Dict[str, torch.Tensor] = {}
        for pkg in packages.values():
            w = pkg["train_samples"] / total
            grads = pkg["_sp_extra"]
            if self.enable_topk_grad:
                grads = self._topk_inactive_filter(grads)
            for name, g in grads.items():
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

    def _topk_inactive_filter(self, grads: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Zero-out all inactive gradients except top-k (by magnitude)."""
        k = int(f_decay(self._sp_t, self.adjust_alpha, self.T_end) * sum(
            (self._sp_mask_dict[n] == 1).sum().item()
            for n in self._sp_mask_dict
        ) / max(len(self._sp_mask_dict), 1))
        filtered = {}
        for name, g in grads.items():
            if name not in self._sp_mask_dict:
                filtered[name] = g
                continue
            g_f = g.clone().float()
            inactive_idx = (self._sp_mask_dict[name].view(-1) == 0).nonzero(as_tuple=False).view(-1)
            if len(inactive_idx) > k:
                layer_k = min(k, len(inactive_idx))
                vals = g_f.abs().view(-1)[inactive_idx]
                threshold = torch.topk(vals, layer_k, largest=True).values.min()
                mask_inactive = g_f.abs() < threshold
                g_f[mask_inactive & (self._sp_mask_dict[name].to(g_f.device) == 0)] = 0.0
            filtered[name] = g_f
        return filtered


class FedMef_Client(spFL_Client):
    """FedMef client — BAE regularization + post-training gradient collection."""

    optional = {
        "lambda_l2": 1e-4,
        "psi": 1e-4,
        "xi": 1.0,
        "gamma": 0.3,
        "enable_dynamic_lowest_k": True,
    }

    def fit(self) -> None:
        from .base import SharedMethods
        SharedMethods._set_worker_seed(self._loader_seed("train"))
        loader = self.load_train_data()
        apply_mask(self.model, self._sp_mask_dict)
        self.model.to(self.device)
        self.model.train()
        init_lr = self.optimizer.param_groups[0]["lr"]
        total_batches = len(loader)

        if not self.enable_dynamic_lowest_k:
            self._penalty_indices = self._compute_penalty_indices()

        for epoch in range(self.epochs):
            for b_idx, (batch_x, batch_y, x_mark, y_mark) in enumerate(loader):
                batch_x = batch_x.to(self.device, dtype=torch.float32)
                batch_y = batch_y.to(self.device, dtype=torch.float32)
                x_mark = x_mark.to(self.device)
                y_mark = y_mark.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
                task_loss = self.loss(pred, batch_y)
                total_loss, l1_loss = self._bae_loss(task_loss)
                self._adjust_lr(init_lr, l1_loss, self.epochs, epoch, b_idx, total_batches)
                total_loss.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
            apply_mask(self.model, self._sp_mask_dict)

        if self.efficiency in ("low", "med"):
            self.model.to("cpu")

    def _compute_penalty_indices(self) -> Dict[str, torch.Tensor]:
        indices = {}
        for name, param in self.model.named_parameters():
            if name in self._sp_mask_dict:
                active = int((self._sp_mask_dict[name] == 1).sum().item())
                k = int(f_decay(self._sp_t, self.gamma, self._sp_T_end) * active)
                _, idx = torch.topk(param.data.abs().flatten(), k, largest=False)
                indices[name] = idx
        return indices

    def _bae_loss(self, task_loss: torch.Tensor):
        low_params = []
        for name, param in self.model.named_parameters():
            if name not in self._sp_mask_dict:
                continue
            if self.enable_dynamic_lowest_k:
                active = int((self._sp_mask_dict[name] == 1).sum().item())
                k = int(f_decay(self._sp_t, self.gamma, self._sp_T_end) * active)
                sorted_vals = param.data.abs().flatten().sort()[0]
                if k > 0 and k < len(sorted_vals):
                    threshold = sorted_vals[k]
                    mask_low = (param.abs() <= threshold).float()
                    low_params.append(param * mask_low)
            else:
                low_params.append(param.flatten()[self._penalty_indices[name]])

        if not low_params:
            return task_loss, torch.tensor(0.0)
        l2 = sum(torch.norm(p, 2) for p in low_params)
        l1 = sum(torch.norm(p, 1) for p in low_params)
        return task_loss + self.lambda_l2 * l2, l1

    def _adjust_lr(self, init_lr: float, l1_loss: torch.Tensor,
                   total_rounds: int, epoch: int, step: int, steps_per_epoch: int) -> None:
        B = steps_per_epoch * total_rounds
        b = step + steps_per_epoch * epoch
        decay = (2 * B - 2 * b) / max(2 * B - b, 1)
        sig = 2 * torch.sigmoid(l1_loss).item() - 1
        adjusted = decay * sig * init_lr
        for pg in self.optimizer.param_groups:
            pg["lr"] = float(min(self.xi, max(pg["lr"], self.psi * adjusted)))

    def package(self, train_time: float) -> Dict[str, Any]:
        result = super().package(train_time)
        result["_sp_extra"] = self._collect_gradients() if self._sp_is_adj else {}
        return result
