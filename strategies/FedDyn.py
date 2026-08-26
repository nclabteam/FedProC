from argparse import Namespace
from collections import OrderedDict
from typing import Any, Dict

import torch

from .tFL import tFL, tFL_Client


class FedDynShared:
    """Dynamic-gradient state operations shared by server and worker."""

    @staticmethod
    def update_dual(
        dual: Any, local_params: Any, global_params: Any, alpha: Any
    ) -> Any:
        return OrderedDict(
            (
                name,
                dual[name] - alpha * (local_params[name] - global_params[name]),
            )
            for name in dual
        )


class FedDyn(FedDynShared, tFL):
    """FedDyn: global FL with per-client dynamic gradient state."""

    optional = {"alpha": 0.1}

    @classmethod
    def args_update(cls, parser: Any) -> None:
        parser.add_argument(
            "--alpha",
            type=float,
            default=None,
            help="FedDyn dynamic-regularization strength",
        )

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        if self.alpha <= 0:
            raise ValueError("FedDyn requires alpha > 0")
        self.server_state = OrderedDict(
            (name, torch.zeros_like(value))
            for name, value in self.public_model_params.items()
        )
        for personal in self.clients_personal_model_params.values():
            personal["old_grad"] = OrderedDict(
                (name, value.clone()) for name, value in self.server_state.items()
            )

    def aggregate_client_updates(self, packages: Any) -> None:
        active_mean = self.mean_models(
            models=[package["regular_model_params"] for package in packages.values()]
        )
        self.server_state = self.mean_models(
            models=[
                personal["old_grad"]
                for personal in self.clients_personal_model_params.values()
            ]
        )
        self._commit_global(
            new_params=OrderedDict(
                (
                    name,
                    active_mean[name] - self.server_state[name] / self.alpha,
                )
                for name in active_mean
            )
        )


class FedDyn_Client(FedDynShared, tFL_Client):
    """FedDyn worker carrying its logical client's dynamic gradient state."""

    def set_parameters(self, package: Dict[str, Any]) -> None:
        model_package = dict(package)
        model_package["personal_model_params"] = {}
        super().set_parameters(package=model_package)
        self._old_grad = OrderedDict(
            (name, value.detach().cpu().clone())
            for name, value in package["personal_model_params"]["old_grad"].items()
        )
        self._global_params = OrderedDict(
            (name, value.detach().cpu().clone())
            for name, value in package["regular_model_params"].items()
        )

    def train_one_epoch(
        self,
        model: Any,
        dataloader: Any,
        optimizer: Any,
        criterion: Any,
        scheduler: Any,
        device: Any,
        offload_after: Any = True,
    ) -> None:
        model.to(device)
        self._move_optimizer_state_to_param_devices(optimizer=optimizer)
        global_params = {
            name: value.to(device) for name, value in self._global_params.items()
        }
        old_grad = {name: value.to(device) for name, value in self._old_grad.items()}
        model.train()
        for batch_x, batch_y, x_mark, y_mark in dataloader:
            optimizer.zero_grad(set_to_none=True)
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            x_mark = x_mark.to(device)
            y_mark = y_mark.to(device)
            outputs = model(batch_x, x_mark=x_mark, y_mark=y_mark)
            criterion(outputs, batch_y).backward()
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        parameter.grad.add_(
                            -old_grad[name]
                            + self.alpha * (parameter.detach() - global_params[name])
                        )
            optimizer.step()
            self.step_scheduler_batch(
                scheduler=scheduler,
                batch_data=batch_x,
            )
        self.step_scheduler_epoch(scheduler=scheduler)
        if offload_after:
            model.to("cpu")

    def package(self) -> Dict[str, Any]:
        package = super().package()
        package["__wire__"] = ("regular_model_params",)
        package.pop("score", None)
        package["personal_model_params"]["old_grad"] = self.update_dual(
            dual=self._old_grad,
            local_params=package["regular_model_params"],
            global_params=self._global_params,
            alpha=self.alpha,
        )
        return package
