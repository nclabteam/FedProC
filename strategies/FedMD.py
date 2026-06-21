import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .hFL import hFL, hFL_Client


class FedMD(hFL):
    """
    FedMD: Heterogenous Federated Learning via Model Distillation.

    Each round:
    1. Server scores each client's model on public data (from clients_personal_model_params)
    2. Server computes consensus = mean of client predictions
    3. Clients receive consensus + public batches via package()
    4. Clients: digest (match consensus on public data) then revisit (local training)

    This avoids self.clients entirely — scores are computed server-side from stored params.
    """

    optional = {
        **hFL.optional,
        "public_dataset": "ETDatasetHour",
        "digest_epochs": 5,
        "revisit_epochs": 1,
        "public_batch_size": 32,
        "public_batch_num": 5,
    }

    compulsory = {"exclude_server_model_processes": True}

    @classmethod
    def args_update(cls, parser):
        hFL.args_update(parser)
        parser.add_argument("--public_dataset", type=str, default=None)
        parser.add_argument("--digest_epochs", type=int, default=None)
        parser.add_argument("--revisit_epochs", type=int, default=None)
        parser.add_argument("--public_batch_size", type=int, default=None)
        parser.add_argument("--public_batch_num", type=int, default=None)

    def __init__(self, configs, times):
        self.consensus = []
        self.public_data = []
        super().__init__(configs, times)
        self.public_loader = self._load_public_dataset(configs)
        self._iter_public = iter(self.public_loader)

    def _load_public_dataset(self, configs):
        import data_factory

        public_args = copy.deepcopy(configs)
        public_args.dataset = self.public_dataset
        dataset_cls = getattr(data_factory, self.public_dataset)
        t = dataset_cls(public_args)
        t.execute()

        all_x, all_y = [], []
        for entry in t.info:
            with np.load(entry["paths"]["train"]) as f:
                all_x.append(f["x"])
                all_y.append(f["y"])
        x = torch.as_tensor(np.concatenate(all_x), dtype=torch.float32)
        y = torch.as_tensor(np.concatenate(all_y), dtype=torch.float32)
        return DataLoader(
            TensorDataset(x, y),
            batch_size=self.public_batch_size,
            shuffle=True,
        )

    def initialize_model(self):
        pass

    def initialize_optimizer(self):
        pass

    def initialize_scheduler(self):
        pass

    def initialize_loss(self):
        pass

    def _load_public_batches(self):
        self.public_data = []
        for _ in range(self.public_batch_num):
            try:
                x, _ = next(self._iter_public)
            except StopIteration:
                self._iter_public = iter(self.public_loader)
                x, _ = next(self._iter_public)
            if len(x) <= 1:
                try:
                    x, _ = next(self._iter_public)
                except StopIteration:
                    self._iter_public = iter(self.public_loader)
                    x, _ = next(self._iter_public)
            self.public_data.append(x.cpu())

    @torch.no_grad()
    def _score_client(self, client_id: int) -> list:
        """Run client's personal model on public batches, return list of prediction tensors."""
        personal = self.clients_personal_model_params[client_id]
        if not personal:
            return [torch.zeros_like(x) for x in self.public_data]
        self.model.load_state_dict(self.public_model_params, strict=False)
        self.model.load_state_dict(personal, strict=False)
        self.model.eval()
        self.model.to(self.device)
        scores = [self.model(x.to(self.device)).clone().cpu() for x in self.public_data]
        self.model.to("cpu")
        return scores

    def _compute_consensus(self):
        all_scores = [self._score_client(cid) for cid in self.selected_clients]
        self.consensus = []
        for batch_scores in zip(*all_scores):
            self.consensus.append(torch.stack(batch_scores, dim=-1).mean(dim=-1).cpu())

    def package(self, client_id: int) -> dict:
        result = super().package(client_id)
        result["consensus"] = self.consensus
        result["public_data"] = self.public_data
        return result

    def train_one_round(self) -> None:
        self._load_public_batches()
        self._compute_consensus()
        packages = self.trainer.train(self.selected_clients)
        self.aggregate_client_updates(packages)

    def aggregate_client_updates(self, packages) -> None:
        # Store each client's trained params (nFL-style — no global aggregation)
        for cid, pkg in packages.items():
            self.clients_personal_model_params[cid].update(pkg["regular_model_params"])


class FedMD_Client(hFL_Client):

    consensus: list = []
    public_data: list = []

    def set_parameters(self, package: dict) -> None:
        self.consensus = package.pop("consensus", [])
        self.public_data = package.pop("public_data", [])
        super().set_parameters(package)

    def fit(self) -> None:
        self._digest()
        self._revisit()

    def _digest(self):
        """Train on public data to match the server consensus predictions."""
        if not self.consensus or not self.public_data:
            return
        self.model.to(self.device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        for _ in range(self.digest_epochs):
            for i, x in enumerate(self.public_data):
                target = self.consensus[i].to(self.device)
                pred = self.model(x.to(self.device))
                loss = F.mse_loss(pred, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def _revisit(self):
        """Standard local training on private data."""
        loader = self.load_train_data()
        offload_after_epoch = self.efficiency == "low"
        for _ in range(self.revisit_epochs):
            self.train_one_epoch(
                model=self.model,
                dataloader=loader,
                optimizer=self.optimizer,
                criterion=self.loss,
                scheduler=self.scheduler,
                device=self.device,
                offload_after=offload_after_epoch,
            )
        if self.efficiency == "med":
            self.model.to("cpu")
