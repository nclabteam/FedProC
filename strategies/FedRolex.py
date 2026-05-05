from typing import Dict, List

import numpy as np
import torch

from .base import Client, Server


class FedRolex(Server):
    """
    FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction.

    The server maintains a global model. Each client receives the full model but
    only trains a rolling subset of output dimensions per layer (gradient masking),
    ensuring all parameters are evenly trained across rounds.

    Reference: Alam et al., "FedRolex: Model-Heterogeneous Federated Learning
    with Rolling Sub-Model Extraction", NeurIPS 2022. arXiv:2211.11614.

    Adaptation note: Original uses architecture-level sub-model extraction
    (reduced layer widths). This implementation uses gradient masking on the
    full model to achieve equivalent rolling coverage without requiring
    variable-width model construction. Task-agnostic.
    """

    optional = {
        "capacity": "1,0.5,0.25,0.125",
    }

    compulsory = {
        "save_local_model": True,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument(
            "--capacity", type=str, default=None,
            help="Comma-separated client capacity ratios (e.g. '1,0.5,0.25,0.125')"
        )

    def __init__(self, configs, times):
        self.round_counter = 0
        super().__init__(configs, times)
        self.capacities = self._parse_capacities()
        self._assign_capacities()

    def _parse_capacities(self) -> List[float]:
        return [float(c) for c in self.capacity.split(",")]

    def _assign_capacities(self):
        for i, client in enumerate(self.clients):
            client.capacity = self.capacities[i % len(self.capacities)]

    def send_to_clients(self):
        for client in self.clients:
            client.receive_from_server({
                "model": self.model,
                "capacity": client.capacity,
                "offset": self.round_counter,
            })

    def aggregate_models(self):
        """Selective aggregation: average each parameter only from clients that trained it."""
        global_state = self.model.state_dict()
        accum = {k: torch.zeros_like(v) for k, v in global_state.items()}
        count = {k: torch.zeros(v.shape[0], dtype=torch.long, device=v.device)
                 for k, v in global_state.items()}

        for client in self.selected_clients:
            client_state = client.send_to_server()["model"]
            capacity = client.capacity
            offset = self.round_counter

            for name, global_param in global_state.items():
                if name not in client_state:
                    continue
                client_param = client_state[name]
                K = global_param.shape[0]
                k = max(1, int(K * capacity))
                start = offset % K
                idx = [(start + i) % K for i in range(k)]

                accum[name][idx] += client_param[idx]
                count[name][idx] += 1

        # Average where count > 0
        for name in global_state:
            mask = count[name] > 0
            for i in torch.where(mask)[0]:
                global_state[name][i] = accum[name][i] / count[name][i]

        self.model.load_state_dict(global_state)
        self.round_counter += 1


class FedRolex_Client(Client):
    """Client that trains only a rolling subset of the global model's output dimensions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capacity = 1.0
        self._offset = 0
        self._window_indices = {}

    def receive_from_server(self, data):
        super().receive_from_server(data)
        self.capacity = data.get("capacity", 1.0)
        self._offset = data.get("offset", 0)
        self._compute_window()

    def _compute_window(self):
        """Pre-compute which output-dimension indices are active for each layer."""
        self._window_indices = {}
        for name, param in self.model.named_parameters():
            if len(param.shape) >= 1:
                K = param.shape[0]
                k = max(1, int(K * self.capacity))
                start = self._offset % K
                idx = [(start + i) % K for i in range(k)]
                self._window_indices[name] = set(idx)

    def train(self):
        """Train only the rolling window of parameters."""
        if self.capacity >= 1.0:
            super().train()
            return

        device = self.device
        self.model.to(device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loader = self.load_train_data()

        for _ in range(self.local_epochs):
            for batch_x, batch_y, x_mark, y_mark in loader:
                batch_x = batch_x.to(device=device, dtype=torch.float32, non_blocking=True)
                batch_y = batch_y.to(device=device, dtype=torch.float32, non_blocking=True)
                x_mark = x_mark.to(device=device, dtype=torch.float32, non_blocking=True)
                y_mark = y_mark.to(device=device, dtype=torch.float32, non_blocking=True)

                optimizer.zero_grad()
                output = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
                loss = self.loss(output, batch_y)
                loss.backward()

                # Mask gradients: zero out parameters outside the rolling window
                for name, param in self.model.named_parameters():
                    if param.grad is not None and name in self._window_indices:
                        mask = torch.ones(param.shape[0], dtype=torch.bool, device=param.device)
                        active = list(self._window_indices[name])
                        inactive = [i for i in range(param.shape[0]) if i not in set(active)]
                        if inactive:
                            mask[inactive] = False
                            # For multi-dim params, expand mask
                            if len(param.shape) > 1:
                                extra_dims = param.shape[1:]
                                mask = mask.view(-1, *([1] * len(extra_dims))).expand_as(param.grad)
                            param.grad[~mask] = 0

                optimizer.step()

        self.model.to("cpu")

    def send_to_server(self):
        return {"model": self.model.state_dict(), "capacity": self.capacity}
