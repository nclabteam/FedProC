import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, TensorDataset

from .tFL import tFL, tFL_Client


class FedLAW(tFL):
    """
    FedLAW: Revisiting Weighted Aggregation in Federated Learning with Neural Networks.

    Server learns optimal aggregation weights λ (per-client softmax) and a global
    scale γ via gradient descent on a small public proxy dataset each round.
    The final global model is: γ · Σ(λᵢ · flat_params_i).

    Reference: Zeng et al., "Revisiting Weighted Aggregation in Federated Learning
    with Neural Networks", ICML 2023. arXiv:2302.10911.
    """

    optional = {
        "public_dataset": "ETDatasetHour",
        "server_epochs": 20,
        "server_lr": 0.005,
        "distill_batch_size": 32,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--public_dataset", type=str, default=None)
        parser.add_argument("--server_epochs", type=int, default=None)
        parser.add_argument("--server_lr", type=float, default=None)
        parser.add_argument("--distill_batch_size", type=int, default=None)

    def __init__(self, configs, times):
        super().__init__(configs, times)
        self.public_loader = self._load_public_dataset(configs)

    def _load_public_dataset(self, configs):
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
        cids = list(packages.keys())
        device = self.device
        self.model.to(device)

        flat_params = torch.stack(
            [packages[cid]["flat_params"].to(device) for cid in cids]
        )  # [N, D]

        scores = torch.tensor(
            [packages[cid]["score"] for cid in cids],
            dtype=torch.float32,
            device=device,
        )
        init_a = torch.log(scores / scores.sum() + 1e-8)
        a = init_a.detach().clone().requires_grad_(True)
        g = torch.zeros(1, device=device, requires_grad=True)

        optimizer = torch.optim.Adam([a, g], lr=self.server_lr)

        param_names = [name for name, _ in self.model.named_parameters()]
        param_shapes = [p.shape for p in self.model.parameters()]
        param_numels = [p.numel() for p in self.model.parameters()]

        self.model.train()
        for _ in range(self.server_epochs):
            for batch_x, batch_y, x_mark, y_mark in self.public_loader:
                batch_x = batch_x.to(device=device, non_blocking=True)
                batch_y = batch_y.to(device=device, non_blocking=True)
                x_mark = x_mark.to(device=device, non_blocking=True)
                y_mark = y_mark.to(device=device, non_blocking=True)

                lam = torch.softmax(a, dim=0)  # [N]
                gamma = torch.exp(g)  # scalar
                merged = gamma * (lam @ flat_params)  # [D]

                param_dict = {}
                offset = 0
                for name, shape, numel in zip(param_names, param_shapes, param_numels):
                    param_dict[name] = merged[offset : offset + numel].view(shape)
                    offset += numel

                pred = torch.func.functional_call(
                    self.model,
                    param_dict,
                    (batch_x,),
                    kwargs={"x_mark": x_mark, "y_mark": y_mark},
                )
                loss = F.mse_loss(pred, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            lam = torch.softmax(a, dim=0)
            gamma = torch.exp(g)
            best_flat = gamma * (lam @ flat_params)
            vector_to_parameters(best_flat, self.model.parameters())

        self.model.to("cpu")
        self._commit_global(
            OrderedDict(
                (k, v.detach().cpu().clone()) for k, v in self.model.named_parameters()
            )
        )


class FedLAW_Client(tFL_Client):
    def package(self, train_time: float) -> dict:
        result = super().package(train_time)
        result["flat_params"] = parameters_to_vector(
            self.model.parameters()
        ).detach().cpu()
        return result
