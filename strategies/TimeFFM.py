from argparse import Namespace
from collections import OrderedDict
from typing import Any

import torch

from utils.parsing import str2bool

from .pFL import pFL, pFL_Client


class TimeFFMShared:
    """Parameter split: ``encoder.*`` is global, ``head.*`` is personal, GPT-2 is frozen."""

    @staticmethod
    def split_names(model: torch.nn.Module) -> tuple[list[str], list[str]]:
        global_names, personal_names = [], []
        for name, param in model.named_parameters():
            if name.startswith("encoder."):
                global_names.append(name)
            elif name.startswith("head."):
                personal_names.append(name)
            elif param.requires_grad:
                raise ValueError(f"unexpected trainable parameter {name}")
        return global_names, personal_names


class TimeFFM(TimeFFMShared, pFL):
    """Time-FFM: LM-empowered federated foundation model for time series forecasting."""

    compulsory = {"model": "TimeFFM"}
    optional = {"backcast_loss": True, "clip": 5.0}

    @classmethod
    def args_update(cls, parser: Any) -> None:
        parser.add_argument(
            "--backcast_loss",
            type=str2bool,
            default=None,
            help="also reconstruct the input window (official code); False = paper Eq. 1",
        )
        parser.add_argument("--clip", type=float, default=None)

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        self.global_names, personal_names = self.split_names(model=self.model)
        for personal in self.clients_personal_model_params.values():
            personal.update(
                {
                    name: self.public_model_params[name].clone()
                    for name in personal_names
                }
            )

    def package(self, client_id: int) -> dict[str, Any]:
        package = super().package(client_id=client_id)
        package["regular_model_params"] = OrderedDict(
            (name, package["regular_model_params"][name]) for name in self.global_names
        )
        package["__wire__"] = ("regular_model_params",)
        return package

    def aggregate_client_updates(
        self, packages: "OrderedDict[int, dict[str, Any]]"
    ) -> None:
        # w^g_{t+1} = (1/N) sum_i w^g_{t,i}  (Alg. 1 line 6, unweighted)
        new_params = OrderedDict(self.public_model_params)
        new_params.update(
            self.mean_models(
                models=[
                    package["regular_model_params"] for package in packages.values()
                ]
            )
        )
        self._commit_global(new_params=new_params)


class TimeFFM_Client(TimeFFMShared, pFL_Client):
    def __init__(self, configs: Namespace, times: int, device: str) -> None:
        super().__init__(configs=configs, times=times, device=device)
        self.regular_params_name, self.personal_params_name = self.split_names(
            model=self.model
        )

    def package(self) -> dict[str, Any]:
        package = super().package()
        package["__wire__"] = ("regular_model_params",)
        return package

    def fit(self) -> None:
        self._set_worker_seed(seed=self._loader_seed(dataset_type="train"))
        loader = self.load_train_data()
        self.initialize_scheduler(steps_per_epoch=len(loader))
        model, device = self.model, self.device
        model.to(device)
        self._move_optimizer_state_to_param_devices(optimizer=self.optimizer)
        model.train()
        c_out = self.output_channels
        for _ in range(self.epochs):
            for batch_x, batch_y, _, _ in loader:
                self.optimizer.zero_grad(set_to_none=True)
                batch_x = batch_x.to(device=device, dtype=torch.float32)
                batch_y = batch_y.to(device=device, dtype=torch.float32)
                outputs = model.forward_full(batch_x)[..., -c_out:]
                if self.backcast_loss:
                    target = torch.cat((batch_x[..., -c_out:], batch_y), dim=1)
                else:
                    outputs, target = outputs[:, -batch_y.shape[1] :], batch_y
                loss = self.loss(outputs, target)
                loss.backward()
                # Official code clips only the head (its encoder clip targets the frozen LM).
                if self.clip:
                    torch.nn.utils.clip_grad_norm_(model.head.parameters(), self.clip)
                self.optimizer.step()
                self.step_scheduler_batch(scheduler=self.scheduler, batch_data=batch_x)
            self.step_scheduler_epoch(scheduler=self.scheduler)
        if self.efficiency != "high":
            model.to("cpu")
