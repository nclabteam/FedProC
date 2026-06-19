import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .base import SharedMethods  # noqa: F401
from .hFL import hFL, hFL_Client


class FedMD(hFL):
    """
    FedMD: Heterogenous Federated Learning via Model Distillation.

    Clients use different model architectures and collaborate via
    knowledge distillation on a shared public dataset.
    """

    optional = {
        **hFL.optional,
        "public_dataset": "ETDatasetHour",
        "digest_epochs": 5,
        "revisit_epochs": 1,
        "public_batch_size": 32,
    }

    compulsory = {"exclude_server_model_processes": True}

    @classmethod
    def args_update(cls, parser):
        hFL.args_update(parser)
        parser.add_argument("--public_dataset", type=str, default=None)
        parser.add_argument("--digest_epochs", type=int, default=None)
        parser.add_argument("--revisit_epochs", type=int, default=None)
        parser.add_argument("--public_batch_size", type=int, default=None)

    def __init__(self, configs, times):
        self.consensus = None
        super().__init__(configs, times)
        self.public_loader = self._load_public_dataset(configs)

    def _load_public_dataset(self, configs):
        """Load public dataset via normal data pipeline."""
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

    def aggregate_models(self):
        pass

    def variables_to_be_sent(self):
        return {"consensus": self.consensus}

    def train_one_round(self, current_round):
        # Communicate: each client predicts on public data
        client_preds = [c.predict_public(self.public_loader) for c in self.clients]
        self.consensus = torch.stack(client_preds).mean(dim=0)

        # Digest + Revisit
        for c in self.clients:
            c.digest(self.consensus, self.public_loader)
            c.revisit()


class FedMD_Client(hFL_Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def receive_from_server(self, data):
        """FedMD doesn't aggregate models — no-op."""

    def predict_public(self, public_loader):
        """Compute predictions on public data."""
        self.model.to(self.device)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch_x, _ in public_loader:
                out = self.model(batch_x.to(self.device))
                preds.append(out.cpu())
        self.model.to("cpu")
        return torch.cat(preds, dim=0)

    def digest(self, consensus, public_loader):
        """Train to match consensus predictions on public data (MSE distillation)."""
        self.model.to(self.device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        idx = 0
        for _ in range(self.digest_epochs):
            for batch_x, _ in public_loader:
                bs = batch_x.size(0)
                target = consensus[idx : idx + bs].to(self.device)
                pred = self.model(batch_x.to(self.device))
                loss = F.mse_loss(pred, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                idx += bs
        self.model.to("cpu")

    def revisit(self):
        """Standard training on private data."""
        for _ in range(self.revisit_epochs):
            SharedMethods.train_one_epoch(
                self.model,
                self.private_loader,
                self.loss,
                self.optimizer,
                self.scheduler,
                self.device,
            )
