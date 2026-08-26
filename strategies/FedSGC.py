# -*- coding: utf-8 -*-
"""FedSGC gradient-congruity sparse training."""

import ast
import json
from argparse import Namespace
from collections.abc import Mapping
from typing import Any, Dict

import torch

from .spFL import spFL, spFL_Client


class FedSGCShared:
    """FedSGC topology and population math."""

    @staticmethod
    def population_scores(
        path_info: str,
        num_clients: int,
        sample_ratio: float,
    ) -> Dict[int, int]:
        """Read client sample counts from the generated dataset manifest."""
        with open(path_info, encoding="utf-8") as stream:
            info = json.load(stream)
        scores: Dict[int, int] = {}
        for client_id in range(num_clients):
            try:
                item = (
                    info[client_id] if isinstance(info, list) else info[str(client_id)]
                )
                shape = ast.literal_eval(item["samples"]["train"]["x"])
                scores[client_id] = max(1, int(shape[0] * sample_ratio))
            except (KeyError, TypeError, ValueError, SyntaxError):
                scores[client_id] = 1
        return scores

    @staticmethod
    def directional_mask(
        parameters: Mapping[str, torch.Tensor],
        gradients: Mapping[str, torch.Tensor],
        mask_dict: Mapping[str, torch.Tensor],
        local_direction: Mapping[str, torch.Tensor],
        global_direction: Mapping[str, torch.Tensor],
        fraction: float,
        lambda_param: float,
    ) -> Dict[str, torch.Tensor]:
        """Apply the paper's prioritized congruity prune-and-grow rules."""
        if not 0 <= lambda_param <= 1:
            raise ValueError("lambda_param must be in [0, 1]")
        updated = spFL.clone_mask(mask_dict=mask_dict)
        for name, mask in mask_dict.items():
            original = mask.flatten().cpu().bool()
            active = original.nonzero(as_tuple=False).flatten()
            inactive = (~original).nonzero(as_tuple=False).flatten()
            count = min(int(fraction * active.numel()), inactive.numel())
            if count <= 0:
                continue
            weights = parameters[name].detach().cpu().abs().flatten()
            gradient = gradients[name].detach().cpu().abs().flatten()
            local = local_direction[name].detach().cpu().flatten()
            global_ = (
                global_direction.get(name, torch.zeros_like(local)).cpu().flatten()
            )

            guided_count = int(lambda_param * count)
            conflict = active[(global_[active] * local[active]) < 0]
            guided_prune = (
                conflict[
                    torch.topk(
                        input=weights[conflict],
                        k=min(guided_count, conflict.numel()),
                        largest=False,
                        sorted=False,
                    ).indices
                ]
                if guided_count and conflict.numel()
                else torch.empty(0, dtype=torch.long)
            )
            prune_pool = active[~torch.isin(active, guided_prune, assume_unique=True)]
            rest_prune_count = min(count - guided_prune.numel(), prune_pool.numel())
            magnitude_prune = (
                prune_pool[
                    torch.topk(
                        input=weights[prune_pool],
                        k=rest_prune_count,
                        largest=False,
                        sorted=False,
                    ).indices
                ]
                if rest_prune_count
                else torch.empty(0, dtype=torch.long)
            )

            agree = inactive[(global_[inactive] * local[inactive]) > 0]
            guided_grow = (
                agree[
                    torch.topk(
                        input=gradient[agree],
                        k=min(guided_count, agree.numel()),
                        largest=True,
                        sorted=False,
                    ).indices
                ]
                if guided_count and agree.numel()
                else torch.empty(0, dtype=torch.long)
            )
            grow_pool = inactive[~torch.isin(inactive, guided_grow, assume_unique=True)]
            rest_grow_count = min(count - guided_grow.numel(), grow_pool.numel())
            magnitude_grow = (
                grow_pool[
                    torch.topk(
                        input=gradient[grow_pool],
                        k=rest_grow_count,
                        largest=True,
                        sorted=False,
                    ).indices
                ]
                if rest_grow_count
                else torch.empty(0, dtype=torch.long)
            )

            flat = original.clone()
            flat[torch.cat([guided_prune, magnitude_prune])] = False
            flat[torch.cat([guided_grow, magnitude_grow])] = True
            updated[name] = flat.view_as(mask)
        return updated


class FedSGC(FedSGCShared, spFL):
    """FedSGC server."""

    optional = {
        "target_density": 0.2,
        "delta_T": 20,
        "adjust_alpha": 0.01,
        "lambda_param": 0.01,
        "A_epochs": None,
    }

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        self._global_direction_map: Dict[str, torch.Tensor] = {}
        scores = self.population_scores(
            path_info=self.path_info,
            num_clients=self.num_clients,
            sample_ratio=self.sample_ratio,
        )
        self._client_scores = {
            client_id: score
            for client_id, score in scores.items()
            if not self.is_new[client_id]
        }

    def package(self, client_id: int) -> Dict[str, Any]:
        package = super().package(client_id=client_id)
        package["_sp_global_direction_map"] = {
            name: direction.clone()
            for name, direction in self._global_direction_map.items()
        }
        if package["_sp_global_direction_map"]:
            package["__wire__"] += ("_sp_global_direction_map",)
        return package

    def aggregate_client_updates(
        self,
        packages: Mapping[int, Dict[str, Any]],
    ) -> None:
        old_model = {
            name: value.clone() for name, value in self.public_model_params.items()
        }
        client_masks = []
        client_models = []
        round_scores = []
        for client_id, package in packages.items():
            score = int(package["score"])
            self._client_scores[client_id] = score
            client_masks.append(
                package.get("_sp_extra", {}).get("mask_dict", self._sp_mask_dict)
            )
            client_models.append(package["regular_model_params"])
            round_scores.append(score)
        residual = max(
            0,
            sum(self._client_scores.values()) - sum(round_scores),
        )
        # Paper Eq. (7): include non-participants' old global sparse model.
        averaged = self.sparse_weighted_mean(
            models=client_models,
            masks=client_masks,
            weights=round_scores,
            fallback_model=old_model,
            fallback_mask=self._sp_mask_dict,
            fallback_weight=float(residual),
        )
        self._commit_global(new_params=averaged)
        candidate_masks = list(client_masks)
        if residual:
            candidate_masks.append(self._sp_mask_dict)
        self._sp_mask_dict = self.magnitude_reprune(
            parameters=self.public_model_params,
            candidate_mask=self.union_masks(masks=candidate_masks),
            layer_densities=self._sp_layer_density,
        )
        self._sp_commit_mask()
        # Paper direction: d_(r+1) = sign(theta_(r+1) - theta_r).
        self._global_direction_map = {
            name: torch.sign(self.public_model_params[name] - old_model[name])
            for name in self._sp_mask_dict
        }


class FedSGC_Client(FedSGCShared, spFL_Client):
    """FedSGC worker."""

    def set_parameters(self, package: Dict[str, Any]) -> None:
        super().set_parameters(package=package)
        self._sp_global_dir = {
            name: direction.detach().cpu().clone()
            for name, direction in package.get("_sp_global_direction_map", {}).items()
        }

    def fit(self) -> None:
        self._set_worker_seed(seed=self._loader_seed(dataset_type="train"))
        loader = self.load_train_data()
        self.initialize_scheduler(steps_per_epoch=len(loader))
        self.apply_mask(model=self.model, mask_dict=self._sp_mask_dict)
        if not self._sp_is_adj:
            self._train_masked_epochs(
                dataloader=loader,
                epochs=self.epochs,
                offload_after_epoch=self.efficiency == "low",
            )
        else:
            before = max(self.epochs - 1, 0) if self.A_epochs is None else self.A_epochs
            if not 0 <= before <= self.epochs:
                raise ValueError("A_epochs must be between zero and epochs")
            initial = {
                name: param.detach().cpu().clone()
                for name, param in self.model.named_parameters()
                if name in self._sp_mask_dict
            }
            self._train_masked_epochs(
                dataloader=loader,
                epochs=before,
                offload_after_epoch=self.efficiency == "low",
            )
            local_direction = {
                name: torch.sign(param.detach().cpu() - initial[name])
                for name, param in self.model.named_parameters()
                if name in self._sp_mask_dict
            }
            gradients = self._collect_gradients()
            fraction = self.f_decay(
                t=self.current_iter,
                alpha=self.adjust_alpha,
                T_end=self.T_end,
            )
            self._sp_mask_dict = self.directional_mask(
                parameters=dict(self.model.named_parameters()),
                gradients=gradients,
                mask_dict=self._sp_mask_dict,
                local_direction=local_direction,
                global_direction=self._sp_global_dir,
                fraction=fraction,
                lambda_param=self.lambda_param,
            )
            self.apply_mask(model=self.model, mask_dict=self._sp_mask_dict)
            self._train_masked_epochs(
                dataloader=loader,
                epochs=self.epochs - before,
                offload_after_epoch=self.efficiency == "low",
            )
        if self.efficiency == "med":
            self.model.to("cpu")

    def package(self) -> Dict[str, Any]:
        extra = (
            {"mask_dict": self.clone_mask(mask_dict=self._sp_mask_dict)}
            if self._sp_is_adj
            else {}
        )
        return self._package_sparse_extra(extra=extra)
