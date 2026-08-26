from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.data import DataLoader

from .pFL import pFL, pFL_Client


class FedSelectShared:
    """Mask operations shared by the FedSelect server and worker."""

    @staticmethod
    def blended_state(
        global_state: Mapping[str, torch.Tensor],
        local_state: Mapping[str, torch.Tensor],
        mask: Mapping[str, torch.Tensor],
    ) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict(
            (
                name,
                torch.where(
                    mask[name].bool(),
                    local_state[name].to(global_value),
                    global_value,
                ),
            )
            for name, global_value in global_state.items()
        )

    @staticmethod
    def global_values(
        state: Mapping[str, torch.Tensor],
        mask: Mapping[str, torch.Tensor],
    ) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict(
            (name, value[~mask[name].bool()].clone()) for name, value in state.items()
        )

    @staticmethod
    def updated_mask(
        mask: Mapping[str, torch.Tensor],
        trained_state: Mapping[str, torch.Tensor],
        initial_state: Mapping[str, torch.Tensor],
        personalization_rate: float,
        personalization_limit: float,
    ) -> OrderedDict[str, torch.Tensor]:
        if not 0 <= personalization_rate <= 1:
            raise ValueError("prune_percent must be between 0 and 1")
        if not 0 <= personalization_limit <= 1:
            raise ValueError("sparsity_bound must be between 0 and 1")

        updated = OrderedDict(
            (name, value.bool().clone()) for name, value in mask.items()
        )
        if personalization_rate == 0:
            return updated

        for name, current in updated.items():
            if "weight" not in name:
                continue
            local_count = int(current.count_nonzero())
            budget = int(personalization_limit * current.numel()) - local_count
            global_indices = (~current).flatten().nonzero().flatten()
            if budget <= 0 or global_indices.numel() == 0:
                continue
            promote = min(
                max(1, round(personalization_rate * global_indices.numel())),
                budget,
            )
            # Paper Algorithm 3: m+ selects the largest p% of
            # abs(u_L - u_0), then m_next = m OR m+.
            delta = (
                (trained_state[name].float() - initial_state[name].float())
                .abs()
                .flatten()
            )
            selected = delta[global_indices].topk(k=promote, sorted=False).indices
            updated[name].view(-1)[global_indices[selected]] = True
        return updated


class FedSelect(FedSelectShared, pFL):
    """FedSelect: grow personalized masks and average remaining global entries."""

    optional = {
        "prune_percent": 0.1,
        "delta_interval": 1,
        "sparsity_bound": 0.5,
    }
    compulsory = {
        "optimizer": "SGD",
        "momentum": 0.0,
        "dampening": 0.0,
        "weight_decay": 0.0,
        "nesterov": False,
    }

    @classmethod
    def args_update(cls, parser: ArgumentParser) -> None:
        parser.add_argument("--prune_percent", type=float, default=None)
        parser.add_argument("--delta_interval", type=int, default=None)
        parser.add_argument("--sparsity_bound", type=float, default=None)

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        if self.delta_interval <= 0:
            raise ValueError("delta_interval must be positive")
        self.updated_mask(
            mask={},
            trained_state={},
            initial_state={},
            personalization_rate=self.prune_percent,
            personalization_limit=self.sparsity_bound,
        )
        initial_mask = {
            name: torch.zeros_like(value, dtype=torch.bool)
            for name, value in self.public_model_params.items()
        }
        initial_state = {
            name: value.detach().cpu().clone()
            for name, value in self.model.named_parameters()
        }
        for personal in self.clients_personal_model_params.values():
            personal["mask"] = {
                name: value.clone() for name, value in initial_mask.items()
            }
            personal["local_model_state"] = {
                name: value.clone() for name, value in initial_state.items()
            }

    def select_clients(self) -> None:
        self._select_all_clients()

    def package(self, client_id: int) -> dict[str, Any]:
        package = super().package(client_id=client_id)
        mask = self.clients_personal_model_params[client_id]["mask"]
        package["global_model_params"] = self.global_values(
            state=self.public_model_params,
            mask=mask,
        )
        package["__wire__"] = ("global_model_params",)
        return package

    def aggregate_client_updates(self, packages: Mapping[int, dict[str, Any]]) -> None:
        models = []
        masks = []
        for package in packages.values():
            models.append(package["regular_model_params"])
            masks.append(package["personal_model_params"]["mask"])
        new_global = OrderedDict()
        for name, current in self.public_model_params.items():
            values = torch.stack([model[name].float() for model in models])
            shared = torch.stack([~mask[name].bool() for mask in masks])
            counts = shared.sum(dim=0)
            # Paper Algorithm 1: theta_g[j] =
            # sum_i((not m_i[j]) theta_i[j]) / sum_i(not m_i[j]).
            summed = (values * shared.to(values.dtype)).sum(dim=0)
            new_global[name] = torch.where(
                counts > 0,
                summed / counts.clamp_min(1).to(summed.dtype),
                current.float(),
            ).to(current.dtype)
        self._commit_global(new_params=new_global)


class FedSelect_Client(FedSelectShared, pFL_Client):
    """FedSelect worker with personalized-then-global alternating SGD passes."""

    def set_parameters(self, package: dict[str, Any]) -> None:
        personal = package["personal_model_params"]
        local_package = dict(package)
        local_package["personal_model_params"] = {}
        super().set_parameters(package=local_package)
        state = self.blended_state(
            global_state=package["regular_model_params"],
            local_state=personal["local_model_state"],
            mask=personal["mask"],
        )
        self.model.load_state_dict(state_dict=state, strict=False)
        self._mask = {
            name: value.bool().clone() for name, value in personal["mask"].items()
        }
        self._initial_state = OrderedDict(
            (name, value.detach().cpu().clone())
            for name, value in self.model.named_parameters()
        )

    def _train_partition(
        self,
        dataloader: DataLoader,
        personalized: bool,
        scheduler: Any | None,
        offload_after: bool,
    ) -> None:
        handles = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            selected = self._mask[name].to(parameter.device)
            if not personalized:
                selected = ~selected
            handles.append(
                parameter.register_hook(
                    lambda gradient, selected=selected: gradient * selected
                )
            )
        try:
            self.train_one_epoch(
                model=self.model,
                dataloader=dataloader,
                optimizer=self.optimizer,
                criterion=self.loss,
                scheduler=scheduler,
                device=self.device,
                offload_after=offload_after,
            )
        finally:
            for handle in handles:
                handle.remove()

    def fit(self) -> None:
        self._set_worker_seed(seed=self._loader_seed(dataset_type="train"))
        dataloader = self.load_train_data()
        self.initialize_scheduler(steps_per_epoch=2 * len(dataloader))
        offload_after = self.efficiency == "low"
        for _ in range(self.epochs):
            # Paper Algorithm 2: first update v, then update u.
            self._train_partition(
                dataloader=dataloader,
                personalized=True,
                scheduler=(self.scheduler if self.scheduler_mode == "batch" else None),
                offload_after=False,
            )
            self._train_partition(
                dataloader=dataloader,
                personalized=False,
                scheduler=self.scheduler,
                offload_after=offload_after,
            )
        if self.efficiency == "med":
            self.model.to(device="cpu")

    def package(self) -> dict[str, Any]:
        package = super().package()
        trained = package["regular_model_params"]
        mask = self._mask
        if self.current_iter % self.delta_interval == 0:
            mask = self.updated_mask(
                mask=mask,
                trained_state=trained,
                initial_state=self._initial_state,
                personalization_rate=self.prune_percent,
                personalization_limit=self.sparsity_bound,
            )
        package["personal_model_params"] = {
            "mask": mask,
            "local_model_state": {
                name: value.clone() for name, value in trained.items()
            },
        }
        package["global_model_params"] = self.global_values(
            state=trained,
            mask=mask,
        )
        package["__wire__"] = ("global_model_params",)
        return package
