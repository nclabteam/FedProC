from argparse import ArgumentParser, Namespace
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from .hFL import hFL, hFL_Client


class FedMDShared:
    """Logit aggregation shared by the FedMD phases."""

    @staticmethod
    def mean_logits(
        client_logits: Sequence[Sequence[torch.Tensor]],
    ) -> list[torch.Tensor]:
        if not client_logits:
            raise ValueError("FedMD requires at least one client")
        batch_count = len(client_logits[0])
        if any(len(logits) != batch_count for logits in client_logits):
            raise ValueError("FedMD clients must score the same public batches")
        # Paper Algorithm 1: f_tilde(x) = (1 / m) sum_k f_k(x).
        return [
            torch.stack(batch_logits, dim=0).mean(dim=0)
            for batch_logits in zip(*client_logits)
        ]


class FedMD(FedMDShared, hFL):
    """FedMD: exchange public-data predictions across heterogeneous models."""

    optional = {
        "public_dataset": "ETDatasetHour",
        "digest_epochs": 5,
        "revisit_epochs": 1,
        "public_batch_size": 32,
        "public_batch_num": 5,
    }
    compulsory = {"exclude_server_model_processes": True}

    @classmethod
    def args_update(cls, parser: ArgumentParser) -> None:
        super().args_update(parser=parser)
        parser.add_argument("--public_dataset", type=str, default=None)
        parser.add_argument("--digest_epochs", type=int, default=None)
        parser.add_argument("--revisit_epochs", type=int, default=None)
        parser.add_argument("--public_batch_size", type=int, default=None)
        parser.add_argument("--public_batch_num", type=int, default=None)

    def __init__(self, configs: Namespace, times: int) -> None:
        self.consensus: list[torch.Tensor] = []
        self.public_data: list[tuple[torch.Tensor, ...]] = []
        self._pretrained = False
        super().__init__(configs=configs, times=times)
        if (
            min(
                self.digest_epochs,
                self.revisit_epochs,
                self.public_batch_size,
                self.public_batch_num,
            )
            <= 0
        ):
            raise ValueError("FedMD epoch and public-batch settings must be positive")
        self.public_loader = self.load_public_data(
            configs=configs,
            dataset_name=self.public_dataset,
            batch_size=self.public_batch_size,
        )
        if len(self.public_loader.dataset) < 2:
            raise ValueError("FedMD public data requires at least two samples")
        self._public_iterator = iter(self.public_loader)

    def select_clients(self) -> None:
        self._select_all_clients()

    def _load_public_batches(self) -> None:
        self.public_data = []
        while len(self.public_data) < self.public_batch_num:
            try:
                batch = next(self._public_iterator)
            except StopIteration:
                self._public_iterator = iter(self.public_loader)
                batch = next(self._public_iterator)
            if len(batch[0]) > 1:
                self.public_data.append(tuple(value.cpu() for value in batch))

    def package(self, client_id: int) -> dict[str, Any]:
        package = super().package(client_id=client_id)
        package["consensus"] = self.consensus
        package["public_data"] = self.public_data
        package["__wire__"] = ("consensus",)
        return package

    def _score_clients(self, pretrain: bool) -> dict[int, list[torch.Tensor]]:
        client_logits = {}
        for client_id in self.selected_clients:
            package = self.package(client_id=client_id)
            package["pretrain"] = pretrain
            package["__wire__"] = ()
            output = self.trainer.worker_for(client_id=client_id).score_public(
                package=package
            )
            self.trainer._write_back(cid=client_id, out=output)
            self.clients_personal_model_params[client_id].update(
                output["regular_model_params"]
            )
            client_logits[client_id] = output["public_logits"]
        return client_logits

    def train_one_round(self) -> dict[int, dict[str, Any]]:
        self._load_public_batches()
        client_logits = self._score_clients(pretrain=not self._pretrained)
        self._pretrained = True
        self.consensus = self.mean_logits(client_logits=list(client_logits.values()))
        packages = self.trainer.train(selected=self.selected_clients)
        self.aggregate_client_updates(packages=packages)
        for client_id, logits in client_logits.items():
            self._uplink_sizes[client_id] = self.get_size(obj=logits)
        return packages


class FedMD_Client(FedMDShared, hFL_Client):
    """Stateless FedMD worker with communicate, digest, and revisit phases."""

    def set_parameters(self, package: dict[str, Any]) -> None:
        local_package = dict(package)
        self.consensus = local_package.pop("consensus", [])
        self.public_data = local_package.pop("public_data", [])
        super().set_parameters(package=local_package)

    def _train_public(
        self,
        targets: Sequence[torch.Tensor],
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        epochs: int,
    ) -> None:
        self.model.to(device=self.device)
        self.model.train()
        optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=self.learning_rate
        )
        for _ in range(epochs):
            for (batch_x, _, x_mark, y_mark), target in zip(self.public_data, targets):
                prediction = self.model(
                    batch_x.to(device=self.device),
                    x_mark=x_mark.to(device=self.device),
                    y_mark=y_mark.to(device=self.device),
                )
                loss = criterion(prediction, target.to(device=self.device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

    def _public_predictions(self) -> list[torch.Tensor]:
        self.model.to(device=self.device)
        self.model.eval()
        with torch.no_grad():
            predictions = [
                self.model(
                    batch_x.to(device=self.device),
                    x_mark=x_mark.to(device=self.device),
                    y_mark=y_mark.to(device=self.device),
                )
                .detach()
                .cpu()
                for batch_x, _, x_mark, y_mark in self.public_data
            ]
        self.model.to(device="cpu")
        return predictions

    def score_public(self, package: dict[str, Any]) -> dict[str, Any]:
        pretrain = bool(package.get("pretrain"))
        self.set_parameters(package=package)
        if pretrain:
            self._train_public(
                targets=[batch_y for _, batch_y, _, _ in self.public_data],
                criterion=self.loss,
                epochs=self.digest_epochs,
            )
            self._revisit()
        output = self.package()
        output["public_logits"] = self._public_predictions()
        output["__wire__"] = ("public_logits",)
        return output

    def fit(self) -> None:
        self._set_worker_seed(seed=self._loader_seed(dataset_type="train"))
        self._digest()
        self._revisit()

    def _digest(self) -> None:
        if not self.consensus or not self.public_data:
            return
        # Paper Algorithm 1 digest: fit f_k(x) to f_tilde(x).
        self._train_public(
            targets=self.consensus,
            criterion=F.mse_loss,
            epochs=self.digest_epochs,
        )

    def _revisit(self) -> None:
        dataloader = self.load_train_data()
        self.initialize_scheduler(
            steps_per_epoch=len(dataloader),
            epochs=self.revisit_epochs,
        )
        offload_after = self.efficiency == "low"
        for _ in range(self.revisit_epochs):
            self.train_one_epoch(
                model=self.model,
                dataloader=dataloader,
                optimizer=self.optimizer,
                criterion=self.loss,
                scheduler=self.scheduler,
                device=self.device,
                offload_after=offload_after,
            )
        if self.efficiency == "med":
            self.model.to(device="cpu")
