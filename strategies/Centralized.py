from collections import OrderedDict

import ray

from .base import SharedMethods
from .tFL import tFL, tFL_Client


class Centralized(tFL):
    """Centralized baseline: server trains directly on all clients' data each round.

    Clients send their data loaders; the server runs gradient steps on them
    sequentially (or in parallel via Ray). No federation — this measures the
    upper-bound performance of a central aggregator with full data access.
    """

    compulsory = {"exclude_server_model_processes": False}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialize_loss()
        self.initialize_optimizer()
        self.initialize_scheduler()

    def aggregate_client_updates(self, packages) -> None:
        if self.parallel:
            futures = []
            for cid, pkg in packages.items():
                train_loader = pkg["train_loader"]
                future = train_one_epoch_remote.remote(
                    model=self.model,
                    dataloader=train_loader,
                    optimizer=self.optimizer,
                    criterion=self.loss,
                    scheduler=self.scheduler,
                    device=self.device,
                    epochs=self.epochs,
                )
                futures.append(future)

            for future in futures:
                model_state, optimizer_state = ray.get(future)
                self.model.load_state_dict(model_state)
                self.optimizer.load_state_dict(optimizer_state)
        else:
            for cid, pkg in packages.items():
                train_loader = pkg["train_loader"]
                for _ in range(self.epochs):
                    self.train_one_epoch(
                        model=self.model,
                        dataloader=train_loader,
                        optimizer=self.optimizer,
                        criterion=self.loss,
                        scheduler=self.scheduler,
                        device=self.device,
                    )

        self._commit_global(
            OrderedDict(
                (k, v.detach().cpu().clone()) for k, v in self.model.named_parameters()
            )
        )

    def evaluate_generalization(self, *args, **kwargs):
        pass  # no personalization in centralized training


class Centralized_Client(tFL_Client):
    def fit(self) -> None:
        pass  # server trains on collected client data; no client-side training

    def package(self) -> dict:
        result = super().package()
        result["train_loader"] = self.load_train_data()
        return result


@ray.remote
def train_one_epoch_remote(
    model, dataloader, optimizer, criterion, scheduler, device, epochs
):
    model.to(device)
    SharedMethods._move_optimizer_state_to_param_devices(optimizer)
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y, x_mark, y_mark in dataloader:
            optimizer.zero_grad()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            x_mark = x_mark.to(device)
            y_mark = y_mark.to(device)
            outputs = model(batch_x, x_mark=x_mark, y_mark=y_mark)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model.state_dict(), optimizer.state_dict()
