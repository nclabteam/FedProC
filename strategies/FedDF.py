import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .hFL import hFL


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
        import data_factory

        public_args = copy.deepcopy(configs)
        public_args.dataset = self.public_dataset
        dataset_cls = getattr(data_factory, self.public_dataset)
        t = dataset_cls(public_args)
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

        return DataLoader(
            TensorDataset(x, y, x_mark, y_mark),
            batch_size=self.distill_batch_size,
            shuffle=True,
        )

    def aggregate_client_updates(self, packages) -> None:
        """Store client params (nFL-style), then distill into server model."""
        for cid, pkg in packages.items():
            self.clients_personal_model_params[cid].update(pkg["regular_model_params"])
        self._distill(packages)

    def _distill(self, packages: dict):
        """Server-side ensemble distillation on public data.

        Teacher = mean of client predictions, computed by loading each client's
        trained params (from packages) into a temporary model copy.
        """
        device = self.device
        temp_model = copy.deepcopy(self.model).to(device)

        self.model.to(device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.distill_lr)

        client_params = [pkg["regular_model_params"] for pkg in packages.values()]

        for _ in range(self.distill_epochs):
            for batch_x, batch_y, x_mark, y_mark in self.public_loader:
                batch_x = batch_x.to(device=device, non_blocking=True)
                x_mark = x_mark.to(device=device, non_blocking=True)
                y_mark = y_mark.to(device=device, non_blocking=True)

                with torch.no_grad():
                    client_preds = []
                    for params in client_params:
                        temp_model.load_state_dict(params, strict=False)
                        temp_model.eval()
                        pred = temp_model(batch_x, x_mark=x_mark, y_mark=y_mark)
                        client_preds.append(pred)
                    teacher = torch.stack(client_preds).mean(dim=0)

                student = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
                loss = F.mse_loss(student, teacher)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.model.to("cpu")
        del temp_model
        self._commit_global(
            OrderedDict(
                (k, v.detach().cpu().clone()) for k, v in self.model.named_parameters()
            )
        )
