# -*- coding: utf-8 -*-
"""FedDST client topology updates and sparse weighted aggregation."""

from collections.abc import Mapping
from typing import Any, Dict

from .spFL import spFL, spFL_Client


class FedDST(spFL):
    """FedDST server."""

    optional = {"A_epochs": None}

    def aggregate_client_updates(
        self,
        packages: Mapping[int, Dict[str, Any]],
    ) -> None:
        if not self._sp_is_adj():
            super().aggregate_client_updates(packages=packages)
            return
        client_masks = []
        client_models = []
        client_scores = []
        for package in packages.values():
            mask = package.get("_sp_extra", {}).get("mask_dict")
            if not mask:
                raise ValueError(
                    "FedDST adjustment uploads require one mask per client"
                )
            client_masks.append(mask)
            client_models.append(package["regular_model_params"])
            client_scores.append(package["score"])
        # Paper Eq. (4): average each coordinate only over clients retaining it.
        averaged = self.sparse_weighted_mean(
            models=client_models,
            masks=client_masks,
            weights=client_scores,
        )
        self._commit_global(new_params=averaged)
        union = self.union_masks(masks=client_masks)
        self._sp_mask_dict = self.magnitude_reprune(
            parameters=self.public_model_params,
            candidate_mask=union,
            layer_densities=self._sp_layer_density,
        )
        self._sp_commit_mask()


class FedDST_Client(spFL_Client):
    """FedDST worker."""

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
            self._train_masked_epochs(
                dataloader=loader,
                epochs=before,
                offload_after_epoch=self.efficiency == "low",
            )
            gradients = self._collect_gradients()
            parameters = dict(self.model.named_parameters())
            # Paper schedule: alpha_r = alpha/2 * (1 + cos((r-1)pi/R_end)).
            fraction = self.f_decay(
                t=max(self.current_iter - 1, 0),
                alpha=self.adjust_alpha,
                T_end=self.T_end,
            )
            self._sp_mask_dict = self.swap_mask(
                parameters=parameters,
                gradients=gradients,
                mask_dict=self._sp_mask_dict,
                fraction=fraction,
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
