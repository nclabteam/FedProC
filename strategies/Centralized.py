from argparse import Namespace
from collections import OrderedDict
from typing import Any

import torch

from .tFL import tFL, tFL_Client


class Centralized(tFL):
    """Oracle baseline that trains one server model over every client's data."""

    compulsory = {"exclude_server_model_processes": False}

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        self.initialize_loss()
        self.initialize_optimizer()
        self.initialize_scheduler()

    def select_clients(self) -> None:
        self.selected_clients = [
            client_id
            for client_id in range(self.num_clients)
            if not self.is_new[client_id]
        ]
        self.current_num_join_clients = len(self.selected_clients)

    @classmethod
    def _train_centralized_epoch(
        cls,
        model: Any,
        dataloaders: Any,
        optimizer: Any,
        criterion: Any,
        scheduler: Any,
        device: Any,
    ) -> None:
        model.to(device)
        Centralized._move_optimizer_state_to_param_devices(optimizer=optimizer)
        model.train()
        for dataloader in dataloaders:
            for batch_x, batch_y, x_mark, y_mark in dataloader:
                optimizer.zero_grad(set_to_none=True)
                batch_x = batch_x.to(device=device, dtype=torch.float32)
                batch_y = batch_y.to(device=device, dtype=torch.float32)
                x_mark = x_mark.to(device=device, dtype=torch.float32)
                y_mark = y_mark.to(device=device, dtype=torch.float32)
                outputs = model(batch_x, x_mark=x_mark, y_mark=y_mark)
                criterion(outputs, batch_y).backward()
                optimizer.step()
                cls.step_scheduler_batch(
                    scheduler=scheduler,
                    batch_data=batch_x,
                )
        cls.step_scheduler_epoch(scheduler=scheduler)

    def aggregate_client_updates(self, packages: Any) -> None:
        dataloaders = [package["train_loader"] for package in packages.values()]
        self.initialize_scheduler(
            steps_per_epoch=sum(len(dataloader) for dataloader in dataloaders)
        )
        for _ in range(self.epochs):
            self._train_centralized_epoch(
                model=self.model,
                dataloaders=dataloaders,
                optimizer=self.optimizer,
                criterion=self.loss,
                scheduler=self.scheduler,
                device=self.device,
            )

        self._commit_global(
            new_params=OrderedDict(
                (k, v.detach().cpu().clone()) for k, v in self.model.named_parameters()
            )
        )

    def _compute_send_mb(self, packages: Any) -> tuple:
        return {}, 0.0


class Centralized_Client(tFL_Client):
    def fit(self) -> None:
        """Leave training to the centralized server."""

    def package(self) -> dict:
        package = super().package()
        package["train_loader"] = self.load_train_data()
        package["regular_model_params"] = OrderedDict()
        package["personal_model_params"] = OrderedDict()
        package["__wire__"] = ()
        return package
