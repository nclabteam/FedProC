import json
from argparse import Namespace
from collections import OrderedDict
from typing import Any

import numpy as np
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

    # Official CosineAnnealingLR(T_max=20, eta_min=1e-6) stepped once per round.
    compulsory = {
        "model": "TimeFFM",
        "epochs": 1,
        "scheduler": "CALR",
        "scheduler_mode": "iteration",
    }
    optional = {
        "backcast_loss": True,
        "clip": 5.0,
        "local_batches": "set",
        "T_max": 20,
        "eta_min": 1e-6,
    }

    @classmethod
    def args_update(cls, parser: Any) -> None:
        parser.add_argument(
            "--backcast_loss",
            type=str2bool,
            default=None,
            help="also reconstruct the input window (official code); False = paper Eq. 1",
        )
        parser.add_argument("--clip", type=float, default=None)
        parser.add_argument(
            "--local_batches",
            type=str,
            default=None,
            choices=["set", "stable"],
            help="set: every client trains sum(batches)/N batches per round; "
            "stable: each client trains its own batch count",
        )

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
        # Official run_avg: drop_last loaders; b_i = sum_j batches_j * (1/N) under
        # `set` (no oversampling in FedProC), or the client's own count under `stable`.
        with open(self.path_info, "r", encoding="utf-8") as f:
            info = json.load(f)
        own = []
        for client in info[: self.num_clients]:
            with np.load(client["paths"]["train"]) as data:
                n = len(data["x"])
            if self.sample_ratio < 1.0:
                n = int(n * self.sample_ratio)
            own.append(n // self.batch_size)
        if self.local_batches == "set":
            self.local_batch_budget = [int(sum(own) / self.num_clients)] * len(own)
        else:
            self.local_batch_budget = own
        # Batches already consumed per client; the official iterator persists
        # across rounds and is only reshuffled after a full pass.
        self.batch_cursor = {i: 0 for i in range(self.num_clients)}

    def package(self, client_id: int) -> dict[str, Any]:
        package = super().package(client_id=client_id)
        package["regular_model_params"] = OrderedDict(
            (name, package["regular_model_params"][name]) for name in self.global_names
        )
        package["batch_start"] = self.batch_cursor[client_id]
        package["local_batches"] = self.local_batch_budget[client_id]
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
        for client_id in packages:
            self.batch_cursor[client_id] += self.local_batch_budget[client_id]
        self._commit_global(new_params=new_params)


class TimeFFM_Client(TimeFFMShared, pFL_Client):
    def __init__(self, configs: Namespace, times: int, device: str) -> None:
        super().__init__(configs=configs, times=times, device=device)
        self.regular_params_name, self.personal_params_name = self.split_names(
            model=self.model
        )

    def set_parameters(self, package: dict[str, Any]) -> None:
        super().set_parameters(package=package)
        self.batch_start = package["batch_start"]
        self.local_batch_budget = package["local_batches"]

    def package(self) -> dict[str, Any]:
        package = super().package()
        package["__wire__"] = ("regular_model_params",)
        return package

    def _pass_order(self, n: int, pass_index: int) -> torch.Tensor:
        """Shuffled sample order of one full pass, reproducible across rounds."""
        generator = torch.Generator()
        if self.seed is None:
            generator.seed()
        else:
            generator.manual_seed(
                self._derive_seed(
                    int(self.seed) + int(self.times), self.id, pass_index, 7
                )
            )
        return torch.randperm(n, generator=generator)

    def fit(self) -> None:
        self._set_worker_seed(seed=self._loader_seed(dataset_type="train"))
        dataset = self.load_train_data().dataset
        n_batches = len(dataset) // self.batch_size  # drop_last=True
        if n_batches == 0:
            raise ValueError("TimeFFM needs at least one full training batch")
        model, device = self.model, self.device
        model.to(device)
        self._move_optimizer_state_to_param_devices(optimizer=self.optimizer)
        model.train()
        c_out = self.output_channels
        order, order_pass = None, -1
        for k in range(self.batch_start, self.batch_start + self.local_batch_budget):
            pass_index, position = divmod(k, n_batches)
            if pass_index != order_pass:
                order = self._pass_order(n=len(dataset), pass_index=pass_index)
                order_pass = pass_index
            index = order[position * self.batch_size : (position + 1) * self.batch_size]
            batch_x, batch_y, _, _ = dataset[index.tolist()]
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
            # https://github.com/yuppielqx/Time-FFM/blob/cd9b90c52366100ab91dbbf15861b501ca7fdd26/engines/client_avg.py#L80
            # torch.nn.utils.clip_grad_norm_(self.client_head.parameters(), self.args.clip)
            # The encoder clip at #L134 targets model.parameters() (the frozen LM).
            if self.clip:
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), self.clip)
            self.optimizer.step()
        self.step_scheduler_epoch(scheduler=self.scheduler)
        if self.efficiency != "high":
            model.to("cpu")
