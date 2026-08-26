# -*- coding: utf-8 -*-
"""FedMef budget-aware extrusion and sparse topology adjustment."""

from collections.abc import Mapping
from typing import Any, Dict

import torch
import torch.nn as nn

from .FedTiny import FedTinyShared
from .spFL import spFL, spFL_Client


class FedMefShared(FedTinyShared):
    """FedMef math shared by its server and worker."""

    @staticmethod
    def extrusion_terms(
        model: nn.Module,
        marked: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return squared L2 penalty and L2 status of marked parameters."""
        squared = torch.zeros((), device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if name in marked:
                indices = marked[name].to(device=param.device)
                squared = squared + param.flatten()[indices].square().sum()
        return squared, squared.sqrt()

    @staticmethod
    def budget_learning_rate(
        initial_lr: float,
        scheduled_lr: float,
        low_norm: torch.Tensor,
        step: int,
        budget: int,
    ) -> float:
        """Return the REX-based BaE learning rate."""
        if budget <= 0:
            raise ValueError("budget must be positive")
        # Paper Eqs. (3-5): beta_t = p(t)(2 sigma(||theta_low||)-1) eta_0,
        # p(t) = (2T-2t)/(2T-t), and mu_t = max(eta_t, beta_t).
        rex = (2 * budget - 2 * step) / (2 * budget - step)
        beta = rex * (2 * torch.sigmoid(low_norm.detach()).item() - 1) * initial_lr
        return max(scheduled_lr, beta)


class FedMef(FedMefShared, spFL):
    """FedMef server."""

    optional = {
        "delta_T": 10,
        "T_end": 300,
        "adjust_alpha": 0.4,
        "lambda_l2": 1e-4,
    }

    def aggregate_client_updates(
        self,
        packages: Mapping[int, Dict[str, Any]],
    ) -> None:
        if not self._sp_is_adj():
            super().aggregate_client_updates(packages=packages)
            return
        names = set(self._sp_mask_dict)
        counts = self.adjustment_counts(
            mask_dict=self._sp_mask_dict,
            names=names,
            current_iter=self.current_iter,
            adjust_alpha=self.adjust_alpha,
            T_end=self.T_end,
        )
        marked = self.lowest_active_indices(
            parameters=self.public_model_params,
            mask_dict=self._sp_mask_dict,
            counts=counts,
        )
        client_models = []
        client_extras = []
        client_scores = []
        for package in packages.values():
            client_models.append(package["regular_model_params"])
            client_extras.append(package.get("_sp_extra", {}))
            client_scores.append(package["score"])
        self._commit_global(
            new_params=self.mean_models(
                models=client_models,
                weights=client_scores,
            )
        )
        # Paper: clients upload only top-K gradients; omitted coordinates are zero.
        gradients = self.mean_sparse_gradients(
            extras=client_extras,
            weights=client_scores,
            mask_dict=self._sp_mask_dict,
            names=names,
        )
        self._sp_mask_dict = self.swap_topology(
            parameters=self.public_model_params,
            gradients=gradients,
            mask_dict=self._sp_mask_dict,
            counts=counts,
            prune_indices=marked,
        )
        self._sp_commit_mask()


class FedMef_Client(FedMefShared, spFL_Client):
    """FedMef worker."""

    def fit(self) -> None:
        if not self._sp_is_adj:
            super().fit()
            return
        self._set_worker_seed(seed=self._loader_seed(dataset_type="train"))
        loader = self.load_train_data()
        self.initialize_scheduler(steps_per_epoch=len(loader))
        names = set(self._sp_mask_dict)
        counts = self.adjustment_counts(
            mask_dict=self._sp_mask_dict,
            names=names,
            current_iter=self.current_iter,
            adjust_alpha=self.adjust_alpha,
            T_end=self.T_end,
        )
        marked = self.lowest_active_indices(
            parameters=dict(self.model.named_parameters()),
            mask_dict=self._sp_mask_dict,
            counts=counts,
        )
        initial_lrs = [group["lr"] for group in self.optimizer.param_groups]
        budget = max(1, self.epochs * len(loader))
        step = 0
        self.apply_mask(model=self.model, mask_dict=self._sp_mask_dict)
        for _ in range(self.epochs):
            self.model.to(self.device)
            self._move_optimizer_state_to_param_devices(optimizer=self.optimizer)
            self.model.train()
            for batch_x, batch_y, x_mark, y_mark in loader:
                scheduled_lrs = [group["lr"] for group in self.optimizer.param_groups]
                self.optimizer.zero_grad(set_to_none=True)
                batch_x = batch_x.to(device=self.device, dtype=torch.float32)
                batch_y = batch_y.to(device=self.device, dtype=torch.float32)
                x_mark = x_mark.to(device=self.device)
                y_mark = y_mark.to(device=self.device)
                prediction = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
                task_loss = self.loss(prediction, batch_y)
                penalty, low_norm = self.extrusion_terms(
                    model=self.model,
                    marked=marked,
                )
                for group, initial_lr, scheduled_lr in zip(
                    self.optimizer.param_groups, initial_lrs, scheduled_lrs
                ):
                    group["lr"] = self.budget_learning_rate(
                        initial_lr=initial_lr,
                        scheduled_lr=scheduled_lr,
                        low_norm=low_norm,
                        step=step,
                        budget=budget,
                    )
                # Paper Eq. (2): L_s = L_task + lambda * ||theta_low||_2^2.
                (task_loss + self.lambda_l2 * penalty).backward()
                for name, param in self.model.named_parameters():
                    if name in self._sp_mask_dict and param.grad is not None:
                        param.grad.mul_(
                            self._sp_mask_dict[name].to(device=param.grad.device)
                        )
                self.optimizer.step()
                for group, scheduled_lr in zip(
                    self.optimizer.param_groups,
                    scheduled_lrs,
                ):
                    group["lr"] = scheduled_lr
                self.step_scheduler_batch(
                    scheduler=self.scheduler,
                    batch_data=batch_x,
                )
                self.apply_mask(model=self.model, mask_dict=self._sp_mask_dict)
                step += 1
            self.step_scheduler_epoch(scheduler=self.scheduler)
            if self.efficiency == "low":
                self.model.to("cpu")
        if self.efficiency == "med":
            self.model.to("cpu")

    def package(self) -> Dict[str, Any]:
        if not self._sp_is_adj:
            return self._package_sparse_extra(extra={})
        names = set(self._sp_mask_dict)
        counts = self.adjustment_counts(
            mask_dict=self._sp_mask_dict,
            names=names,
            current_iter=self.current_iter,
            adjust_alpha=self.adjust_alpha,
            T_end=self.T_end,
        )
        gradients = self._collect_gradients(names=names)
        return self._package_sparse_extra(
            extra=self.topk_inactive_gradients(
                gradients=gradients,
                mask_dict=self._sp_mask_dict,
                counts=counts,
            )
        )
