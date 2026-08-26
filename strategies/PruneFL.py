# -*- coding: utf-8 -*-
"""PruneFL adaptive risk-reduction-per-round-time pruning."""

from argparse import Namespace
from collections.abc import Mapping
from typing import Any, Dict

import torch

from .spFL import spFL, spFL_Client


class PruneFLShared:
    """Paper-specific adaptive architecture search."""

    @staticmethod
    def adaptive_mask(
        parameters: Mapping[str, torch.Tensor],
        squared_gradients: Mapping[str, torch.Tensor],
        mask_dict: Mapping[str, torch.Tensor],
        max_active: int,
        max_prune_fraction: float,
        time_constant: float,
    ) -> Dict[str, torch.Tensor]:
        """Maximize approximate risk reduction per normalized round time."""
        if not 0 <= max_prune_fraction <= 1:
            raise ValueError("max_prune_fraction must be in [0, 1]")
        if time_constant <= 0:
            raise ValueError("time_constant must be positive")

        coefficient = 1.0
        protected: Dict[str, torch.Tensor] = {}
        candidates: Dict[str, torch.Tensor] = {}
        candidate_scores = []
        protected_score = torch.tensor(0.0, dtype=torch.float64)
        protected_count = 0

        for name, mask in mask_dict.items():
            flat_mask = mask.detach().cpu().bool().flatten()
            active = flat_mask.nonzero(as_tuple=False).flatten()
            removable = min(int(max_prune_fraction * active.numel()), active.numel())
            keep = active.numel() - removable
            weights = parameters[name].detach().cpu().abs().flatten()
            protected[name] = (
                active[
                    torch.topk(
                        input=weights[active],
                        k=keep,
                        largest=True,
                        sorted=False,
                    ).indices
                ]
                if keep > 0
                else torch.empty(0, dtype=torch.long)
            )
            protected_mask = torch.zeros_like(flat_mask)
            protected_mask[protected[name]] = True
            candidates[name] = (~protected_mask).nonzero(as_tuple=False).flatten()
            scores = squared_gradients[name].detach().cpu().double().flatten()
            candidate_scores.append(scores[candidates[name]])
            protected_score += scores[protected[name]].sum()
            protected_count += protected[name].numel()

        for name, gradient in squared_gradients.items():
            if name not in mask_dict:
                protected_score += gradient.detach().cpu().double().sum()

        all_scores = torch.cat(candidate_scores) if candidate_scores else torch.empty(0)
        selected_global = torch.empty(0, dtype=torch.long)
        if all_scores.numel() > 0 and protected_count < max_active:
            sorted_scores, order = torch.sort(all_scores, descending=True)
            before = torch.cat(
                [torch.zeros(1, dtype=torch.float64), sorted_scores.cumsum(0)[:-1]]
            )
            counts = torch.arange(sorted_scores.numel(), dtype=torch.float64)
            # Paper objective: Gamma(M) = sum_{j in M} g_j^2 /
            # (c + sum_{j in M} t_j), with equal normalized t_j here.
            threshold = (protected_score + before) / (
                time_constant + coefficient * (protected_count + counts)
            )
            accepted = (
                (sorted_scores / coefficient >= threshold)
                .to(torch.int64)
                .cumprod(dim=0)
            )
            count = min(int(accepted.sum()), max_active - protected_count)
            selected_global = order[:count]

        result: Dict[str, torch.Tensor] = {}
        offset = 0
        for name, mask in mask_dict.items():
            count = candidates[name].numel()
            local = (
                selected_global[
                    (selected_global >= offset) & (selected_global < offset + count)
                ]
                - offset
            )
            flat = torch.zeros(mask.numel(), dtype=torch.bool)
            flat[protected[name]] = True
            flat[candidates[name][local]] = True
            result[name] = flat.view_as(mask)
            offset += count
        return result


class PruneFL(PruneFLShared, spFL):
    """Server-side PruneFL architecture search."""

    optional = {
        "delta_T": 50,
        "adjust_alpha": 0.3,
        "adjust_half_life": 10_000,
        "time_constant": 1.0,
    }

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)

    def _sp_is_adj(self) -> bool:
        return self.current_iter > 0 and self.current_iter % self.delta_T == 0

    def _sp_update_mask(self, packages: Mapping[int, Dict[str, Any]]) -> None:
        gradients = []
        scores = []
        for package in packages.values():
            if package.get("_sp_extra"):
                gradients.append(package["_sp_extra"])
                scores.append(package["score"])
        if not gradients:
            return
        averaged = self.mean_models(
            models=gradients,
            weights=scores,
        )
        # Official schedule: max prune difference decays by a 10k-round half-life.
        fraction = self.adjust_alpha * 0.5 ** (
            self.current_iter / self.adjust_half_life
        )
        max_active = sum(mask.numel() for mask in self._sp_mask_dict.values())
        self._sp_mask_dict = self.adaptive_mask(
            parameters=self.public_model_params,
            squared_gradients=averaged,
            mask_dict=self._sp_mask_dict,
            max_active=max_active,
            max_prune_fraction=fraction,
            time_constant=self.time_constant,
        )


class PruneFL_Client(PruneFLShared, spFL_Client):
    """Worker that reports squared dense gradient evidence when requested."""

    def fit(self) -> None:
        self._set_worker_seed(seed=self._loader_seed(dataset_type="train"))
        loader = self.load_train_data()
        self.initialize_scheduler(steps_per_epoch=len(loader))
        self.apply_mask(model=self.model, mask_dict=self._sp_mask_dict)
        accumulated: Dict[str, torch.Tensor] = {}
        steps = 0
        for _ in range(self.epochs):
            self.model.to(self.device)
            self._move_optimizer_state_to_param_devices(optimizer=self.optimizer)
            self.model.train()
            for batch_x, batch_y, x_mark, y_mark in loader:
                self.optimizer.zero_grad(set_to_none=True)
                batch_x = batch_x.to(device=self.device, dtype=torch.float32)
                batch_y = batch_y.to(device=self.device, dtype=torch.float32)
                x_mark = x_mark.to(device=self.device)
                y_mark = y_mark.to(device=self.device)
                prediction = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
                self.loss(prediction, batch_y).backward()
                if self._sp_is_adj:
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            square = param.grad.detach().cpu().square()
                            accumulated[name] = (
                                accumulated.get(name, torch.zeros_like(square)) + square
                            )
                    steps += 1
                for name, param in self.model.named_parameters():
                    if name in self._sp_mask_dict and param.grad is not None:
                        param.grad.mul_(
                            self._sp_mask_dict[name].to(device=param.grad.device)
                        )
                self.optimizer.step()
                self.step_scheduler_batch(
                    scheduler=self.scheduler,
                    batch_data=batch_x,
                )
                self.apply_mask(model=self.model, mask_dict=self._sp_mask_dict)
            self.step_scheduler_epoch(scheduler=self.scheduler)
            if self.efficiency == "low":
                self.model.to("cpu")
        if self.efficiency == "med":
            self.model.to("cpu")
        self._prunefl_squared = (
            {name: value / steps for name, value in accumulated.items()}
            if steps
            else {}
        )

    def package(self) -> Dict[str, Any]:
        return self._package_sparse_extra(extra=self._prunefl_squared)
