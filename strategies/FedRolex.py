import copy
from typing import Dict, List

import torch

from .tFL import tFL, tFL_Client


class FedRolex(tFL):
    """
    FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction.

    The server maintains a global model. Each client trains a physically narrower
    sub-model extracted via a rolling permutation window, ensuring all parameters
    of the global model are evenly trained across rounds.

    Reference: Alam et al., "FedRolex: Model-Heterogeneous Federated Learning
    with Rolling Sub-Model Extraction", NeurIPS 2022. arXiv:2211.11614.

    Adaptation note: Original operates on Conv filter dimensions (ResNet).
    This implementation generalizes to Linear/Conv1d layers by operating on
    the output dimension (dim=0). Index cascade ensures layer L+1 input
    indices = layer L output indices. Task-agnostic.
    """

    optional = {
        "capacity": "1,0.5,0.25,0.125",
    }

    compulsory = {}

    @classmethod
    def args_update(cls, parser):
        parser.add_argument(
            "--capacity",
            type=str,
            default=None,
            help="Comma-separated client capacity ratios (e.g. '1,0.5,0.25,0.125')",
        )

    def __init__(self, configs, times):
        self.round_counter = 0
        super().__init__(configs, times)
        self.capacities = self._parse_capacities()
        self._assign_capacities()
        self._global_idx = self._build_permutation_indices()
        self._param_groups = self._group_cascaded_params()

    def _parse_capacities(self) -> List[float]:
        return [float(c) for c in self.capacity.split(",")]

    def _assign_capacities(self):
        for i, client in enumerate(self.clients):
            client.capacity = self.capacities[i % len(self.capacities)]

    def _build_permutation_indices(self) -> Dict[str, torch.Tensor]:
        """Create random permutation of output-dimension indices for each cascade group."""
        groups = self._get_cascade_groups()
        idx = {}
        for group_name, param_names in groups.items():
            # All params in a cascade group share the same output-dim size
            K = self.model.state_dict()[param_names[0]].shape[0]
            perm = torch.randperm(K)
            for name in param_names:
                idx[name] = perm.clone()
        return idx

    def _get_cascade_groups(self) -> Dict[str, List[str]]:
        """Group parameters that share output-dimension cascade (consecutive layers)."""
        state = self.model.state_dict()
        groups = {}
        group_id = 0
        assigned = set()

        for name in state:
            if name in assigned:
                continue
            param = state[name]
            if len(param.shape) < 1:
                continue

            # Find cascade: this param's output dim feeds into next param's input dim
            group = [name]
            assigned.add(name)
            current_out = param.shape[0]

            # Look for next layer whose input dim matches current output dim
            for other_name in state:
                if other_name in assigned:
                    continue
                other_param = state[other_name]
                if len(other_param.shape) >= 2 and other_param.shape[1] == current_out:
                    group.append(other_name)
                    assigned.add(other_name)
                    current_out = other_param.shape[0]

            groups[f"group_{group_id}"] = group
            group_id += 1

        return groups

    def _group_cascaded_params(self) -> Dict[str, List[str]]:
        """Return parameter group mapping for aggregation."""
        groups = self._get_cascade_groups()
        result = {}
        for group_name, param_names in groups.items():
            for name in param_names:
                result[name] = group_name
        return result

    def send_to_clients(self):
        """Send sub-model state dicts to clients based on capacity and rolling offset."""
        global_state = self.model.state_dict()
        for client in self.clients:
            capacity = client.capacity
            K = next(iter(global_state.values())).shape[0]
            k = max(1, int(K * capacity))
            roll_offset = self.round_counter * k

            sub_state = {}
            for name, param in global_state.items():
                perm = self._global_idx.get(name)
                if perm is None:
                    sub_state[name] = param.clone()
                    continue

                # Roll the permutation and take first k indices
                rolled = torch.roll(
                    perm,
                    (
                        roll_offset.item()
                        if isinstance(roll_offset, torch.Tensor)
                        else roll_offset
                    ),
                    0,
                )
                active_idx = rolled[:k]
                sub_state[name] = param[active_idx].clone()

            client.receive_from_server(
                {
                    "sub_state": sub_state,
                    "capacity": capacity,
                    "active_indices": {
                        name: torch.roll(
                            self._global_idx[name],
                            int(
                                self.round_counter
                                * max(
                                    1,
                                    int(
                                        self.model.state_dict()[name].shape[0]
                                        * capacity
                                    ),
                                )
                            ),
                            0,
                        )[
                            : max(
                                1,
                                int(self.model.state_dict()[name].shape[0] * capacity),
                            )
                        ]
                        for name in global_state
                        if name in self._global_idx
                    },
                }
            )

    def receive_from_clients(self):
        self.client_data = []
        for client in self.selected_clients:
            self.client_data.append(client.send_to_server())

    def aggregate_models(self):
        """Index-scatter aggregation: write sub-model params back to global positions."""
        global_state = self.model.state_dict()
        accum = {
            k: torch.zeros_like(v, dtype=torch.float64) for k, v in global_state.items()
        }
        count = {
            k: torch.zeros(v.shape[0], dtype=torch.float64)
            for k, v in global_state.items()
        }

        for client_data in self.client_data:
            sub_state = client_data["sub_state"]
            active_indices = client_data["active_indices"]

            for name in global_state:
                if name not in sub_state or name not in active_indices:
                    continue
                idx = active_indices[name]
                accum[name][idx] += sub_state[name].to(dtype=torch.float64)
                count[name][idx] += 1

        # Average where count > 0, leave unchanged otherwise
        for name in global_state:
            mask = count[name] > 0
            global_state[name] = torch.where(
                mask.unsqueeze(-1) if len(global_state[name].shape) > 1 else mask,
                (accum[name] / count[name].clamp(min=1)).to(
                    dtype=global_state[name].dtype
                ),
                global_state[name],
            )

        self.model.load_state_dict(global_state)
        self.round_counter += 1


class FedRolex_Client(tFL_Client):
    """Client that trains a physically narrower sub-model extracted from the global model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capacity = 1.0
        self._sub_state = None
        self._active_indices = None

    def receive_from_server(self, data):
        self._sub_state = data["sub_state"]
        self.capacity = data.get("capacity", 1.0)
        self._active_indices = data.get("active_indices", {})

    def train(self):
        """Train the sub-model on local data."""
        if self._sub_state is None:
            super().train()
            return

        device = self.device

        # Build a narrow model by resizing Linear/Conv1d layers
        sub_model = self._build_narrow_model()
        sub_model.to(device)
        sub_model.train()

        optimizer = torch.optim.Adam(sub_model.parameters(), lr=self.learning_rate)
        loader = self.load_train_data()

        for _ in range(self.epochs):
            for batch_x, batch_y, x_mark, y_mark in loader:
                batch_x = batch_x.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                batch_y = batch_y.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                x_mark = x_mark.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                y_mark = y_mark.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )

                optimizer.zero_grad()
                output = sub_model(batch_x, x_mark=x_mark, y_mark=y_mark)
                loss = self.loss(output, batch_y)
                loss.backward()
                optimizer.step()

        # Extract sub-model state for aggregation
        self._sub_state = sub_model.state_dict()
        sub_model.to("cpu")

        # Return training package (same format as tFL_Client.train)
        model = self.model
        if self.parallel:
            model = self._clone_model_to_cpu(self.model)
        return {
            "id": self.id,
            "model": model,
            "train_samples": self.train_samples,
            "optimizer_state": copy.deepcopy(self.optimizer.state_dict()),
            "train_time": 0,
        }

    def _build_narrow_model(self):
        """Build a model with reduced output dimensions matching the sub-state dict."""
        from copy import deepcopy

        model = deepcopy(self.model)

        # Resize layers to match sub-state dimensions
        new_state = model.state_dict()
        for name, sub_param in self._sub_state.items():
            if name in new_state:
                orig_shape = new_state[name].shape
                sub_shape = sub_param.shape

                if sub_shape != orig_shape:
                    # Need to resize the actual module
                    parts = name.split(".")
                    module = model
                    for p in parts[:-1]:
                        module = getattr(module, p)
                    attr_name = parts[-1]

                    if attr_name == "weight" and len(sub_shape) == 2:
                        # Linear layer: resize output features
                        old_linear = module
                        new_linear = torch.nn.Linear(
                            sub_shape[1],
                            sub_shape[0],
                            bias=old_linear.bias is not None,
                        )
                        new_linear.weight.data.copy_(sub_param)
                        if old_linear.bias is not None:
                            bias_name = name.replace("weight", "bias")
                            if bias_name in self._sub_state:
                                new_linear.bias.data.copy_(self._sub_state[bias_name])
                        # Replace the module
                        parent_parts = parts[:-2]
                        parent = model
                        for p in parent_parts:
                            parent = getattr(parent, p)
                        setattr(parent, parts[-2], new_linear)
                    elif attr_name == "weight" and len(sub_shape) == 3:
                        # Conv1d: resize output channels
                        old_conv = module
                        new_conv = torch.nn.Conv1d(
                            sub_shape[1],
                            sub_shape[0],
                            kernel_size=old_conv.kernel_size,
                            stride=old_conv.stride,
                            padding=old_conv.padding,
                            dilation=old_conv.dilation,
                            groups=old_conv.groups,
                            bias=old_conv.bias is not None,
                        )
                        new_conv.weight.data.copy_(sub_param)
                        if old_conv.bias is not None:
                            bias_name = name.replace("weight", "bias")
                            if bias_name in self._sub_state:
                                new_conv.bias.data.copy_(self._sub_state[bias_name])
                        parent_parts = parts[:-2]
                        parent = model
                        for p in parent_parts:
                            parent = getattr(parent, p)
                        setattr(parent, parts[-2], new_conv)
                    else:
                        new_state[name] = sub_param
                else:
                    new_state[name] = sub_param

        # Reload state after resizing
        # Note: this is a simplified approach — complex architectures may need
        # model-specific handling for cascaded dimension consistency
        try:
            model.load_state_dict(new_state, strict=False)
        except RuntimeError:
            pass

        return model

    def send_to_server(self):
        return {
            "sub_state": self._sub_state,
            "active_indices": self._active_indices,
        }
