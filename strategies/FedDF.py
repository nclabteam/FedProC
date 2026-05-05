import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .base import Server
from .hFL import hFL, hFL_Client


class FedDF(hFL):
    """
    FedDF: Ensemble Distillation for Robust Model Fusion in Federated Learning.

    Server-side ensemble distillation: after FedAvg aggregation, the server
    refines the global model by distilling from the ensemble of client models
    on unlabeled/public data.

    Reference: Lin et al., "Ensemble Distillation for Robust Model Fusion
    in Federated Learning", NeurIPS 2020. arXiv:2006.07242.

    Adaptation note: The original paper uses softmax + KL-divergence for
    classification (CV/NLP). This implementation uses MSE loss for
    time-series forecasting (regression). Please use with caution — the
    distillation dynamics may differ from the classification setting.
    """

    optional = {
        **hFL.optional,
        "public_dataset": "ETDatasetHour",
        "distill_epochs": 5,
        "distill_batch_size": 32,
        "distill_lr": 1e-3,
    }

    compulsory = {
        **hFL.compulsory,
    }

    @classmethod
    def args_update(cls, parser):
        hFL.args_update(parser)
        parser.add_argument("--public_dataset", type=str, default=None)
        parser.add_argument("--distill_epochs", type=int, default=None)
        parser.add_argument("--distill_batch_size", type=int, default=None)
        parser.add_argument("--distill_lr", type=float, default=None)

    def __init__(self, configs, times):
        super().__init__(configs, times)
        self.public_loader = self._load_public_dataset(configs)

    def _load_public_dataset(self, configs):
        """Load public dataset for server-side distillation."""
        from data_factory import BaseDataset

        public_args = copy.deepcopy(configs)
        public_args.dataset = self.public_dataset
        t = BaseDataset(public_args)
        t.execute()

        all_x, all_y, all_x_mark, all_y_mark = [], [], [], []
        for entry in t.info:
            with np.load(entry["paths"]["train"]) as f:
                all_x.append(f["x"])
                all_y.append(f["y"])
                all_x_mark.append(f["x_mark"])
                all_y_mark.append(f["y_mark"])
        x = torch.as_tensor(np.concatenate(all_x), dtype=torch.float32)
        y = torch.as_tensor(np.concatenate(all_y), dtype=torch.float32)
        x_mark = torch.as_tensor(np.concatenate(all_x_mark), dtype=torch.float32)
        y_mark = torch.as_tensor(np.concatenate(all_y_mark), dtype=torch.float32)

        scaler = t.get_scaler()
        x = torch.as_tensor(scaler.transform(x), dtype=torch.float32)
        y = torch.as_tensor(scaler.transform(y), dtype=torch.float32)

        return DataLoader(
            TensorDataset(x, y, x_mark, y_mark),
            batch_size=self.distill_batch_size,
            shuffle=True,
        )

    def aggregate_models(self):
        """FedAvg aggregation followed by server-side ensemble distillation."""
        super().aggregate_models()
        self._distill()

    def _distill(self):
        """Server-side ensemble distillation on public data.

        The server model is refined to match the averaged predictions
        of all client models on the public dataset.
        """
        device = self.clients[0].device
        self.model.to(device)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.distill_lr)

        for epoch in range(self.distill_epochs):
            for batch_x, batch_y, x_mark, y_mark in self.public_loader:
                batch_x = batch_x.to(device=device, non_blocking=True)
                x_mark = x_mark.to(device=device, non_blocking=True)
                y_mark = y_mark.to(device=device, non_blocking=True)

                # Ensemble: average client predictions (teacher)
                with torch.no_grad():
                    client_preds = []
                    for client in self.selected_clients:
                        client.model.to(device)
                        client.model.eval()
                        pred = client.model(batch_x, x_mark=x_mark, y_mark=y_mark)
                        client_preds.append(pred)
                    teacher = torch.stack(client_preds).mean(dim=0)

                # Student: server model prediction
                student = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)

                # MSE distillation (regression)
                loss = F.mse_loss(student, teacher)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.model.to("cpu")
