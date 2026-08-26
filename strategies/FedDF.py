import copy
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from .hFL import hFL, hFL_Client


class FedDF(hFL):
    """FedDF: distill all received models into each architecture prototype."""

    optional = {
        "public_dataset": "ETDatasetHour",
        "distill_epochs": 5,
        "distill_batch_size": 128,
        "distill_lr": 1e-3,
    }

    @classmethod
    def args_update(cls, parser: ArgumentParser) -> None:
        super().args_update(parser=parser)
        parser.add_argument("--public_dataset", type=str, default=None)
        parser.add_argument("--distill_epochs", type=int, default=None)
        parser.add_argument("--distill_batch_size", type=int, default=None)
        parser.add_argument("--distill_lr", type=float, default=None)

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        if (
            min(
                self.distill_epochs,
                self.distill_batch_size,
                self.distill_lr,
            )
            <= 0
        ):
            raise ValueError("FedDF distillation settings must be positive")
        self.public_loader = self.load_public_data(
            configs=configs,
            dataset_name=self.public_dataset,
            batch_size=self.distill_batch_size,
        )
        self.prototype_model_params = {
            prototype_id: copy.deepcopy(
                self.clients_personal_model_params[client_ids[0]]
            )
            for prototype_id, client_ids in self.trainer.prototype_clients.items()
        }

    def package(self, client_id: int) -> dict[str, Any]:
        package = super().package(client_id=client_id)
        prototype_id = self.trainer.client_prototypes[client_id]
        package["regular_model_params"] = copy.deepcopy(
            self.prototype_model_params[prototype_id]
        )
        package["__wire__"] = ("regular_model_params",)
        return package

    def aggregate_client_updates(self, packages: Mapping[int, dict[str, Any]]) -> None:
        if not packages:
            raise ValueError("FedDF requires at least one client update")
        grouped: dict[int, list[dict[str, Any]]] = {
            prototype_id: [] for prototype_id in self.trainer.prototype_workers
        }
        teacher_models = []
        for client_id, package in packages.items():
            grouped[self.trainer.client_prototypes[client_id]].append(package)
            teacher = copy.deepcopy(self.trainer.worker_for(client_id=client_id).model)
            teacher.load_state_dict(
                state_dict=package["regular_model_params"], strict=False
            )
            teacher.to(device=self.device)
            teacher.eval()
            teacher_models.append(teacher)

        students = {}
        optimizers = {}
        for prototype_id, prototype_packages in grouped.items():
            if not prototype_packages:
                continue
            # Paper Algorithm 3, pseudocode step 11: sample-weighted prototype start.
            initial = self.mean_models(
                models=[
                    package["regular_model_params"] for package in prototype_packages
                ],
                weights=[package["score"] for package in prototype_packages],
            )
            student = self.trainer.prototype_workers[prototype_id].model
            student.load_state_dict(state_dict=initial, strict=False)
            student.to(device=self.device)
            student.train()
            students[prototype_id] = student
            optimizers[prototype_id] = torch.optim.Adam(
                params=student.parameters(), lr=self.distill_lr
            )

        for _ in range(self.distill_epochs):
            for batch_x, _, x_mark, y_mark in self.public_loader:
                batch_x = batch_x.to(device=self.device, non_blocking=True)
                x_mark = x_mark.to(device=self.device, non_blocking=True)
                y_mark = y_mark.to(device=self.device, non_blocking=True)
                with torch.no_grad():
                    # Paper AVGLOGITS, adapted from softmax-KL to TSF outputs.
                    teacher = torch.stack(
                        [
                            model(
                                batch_x,
                                x_mark=x_mark,
                                y_mark=y_mark,
                            )
                            for model in teacher_models
                        ],
                        dim=0,
                    ).mean(dim=0)
                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
                losses = [
                    F.mse_loss(
                        student(
                            batch_x,
                            x_mark=x_mark,
                            y_mark=y_mark,
                        ),
                        teacher,
                    )
                    for student in students.values()
                ]
                torch.stack(losses).sum().backward()
                for optimizer in optimizers.values():
                    optimizer.step()

        for prototype_id, student in students.items():
            student.to(device="cpu")
            state = OrderedDict(
                (name, value.detach().cpu().clone())
                for name, value in student.state_dict().items()
            )
            self.prototype_model_params[prototype_id] = state
            for client_id in self.trainer.prototype_clients[prototype_id]:
                self.clients_personal_model_params[client_id] = copy.deepcopy(state)


class FedDF_Client(hFL_Client):
    """Stateless FedDF worker; full local models cross the wire."""

    def package(self) -> dict[str, Any]:
        package = super().package()
        package["__wire__"] = ("regular_model_params", "score")
        return package
