import copy
from collections import OrderedDict
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
        self._global_idx = self._build_permutation_indices()
        self._param_groups = self._group_cascaded_params()

    def _parse_capacities(self) -> List[float]:
        return [float(c) for c in self.capacity.split(",")]

    def _capacity_for_cid(self, client_id: int) -> float:
        return self.capacities[client_id % len(self.capacities)]

    def _build_permutation_indices(self) -> Dict[str, torch.Tensor]:
        """Create random permutation of output-dimension indices for each cascade group."""
        groups = self._get_cascade_groups()
        idx = {}
        for group_name, param_names in groups.items():
            K = self.public_model_params[param_names[0]].shape[0]
            perm = torch.randperm(K)
            for name in param_names:
                idx[name] = perm.clone()
        return idx

    def _get_cascade_groups(self) -> Dict[str, List[str]]:
        """Group parameters that share output-dimension cascade (consecutive layers)."""
        state = dict(self.public_model_params)
        groups = {}
        group_id = 0
        assigned = set()

        for name in state:
            if name in assigned:
                continue
            param = state[name]
            if len(param.shape) < 1:
                continue

            group = [name]
            assigned.add(name)
            current_out = param.shape[0]

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
        groups = self._get_cascade_groups()
        result = {}
        for group_name, param_names in groups.items():
            for name in param_names:
                result[name] = group_name
        return result

    def package(self, client_id: int) -> dict:
        pkg = super().package(client_id)
        capacity = self._capacity_for_cid(client_id)
        global_state = dict(self.public_model_params)

        sub_state = {}
        active_indices = {}
        for name, param in global_state.items():
            perm = self._global_idx.get(name)
            if perm is None:
                sub_state[name] = param.clone()
                continue
            k = max(1, int(param.shape[0] * capacity))
            roll_offset = self.round_counter * k
            rolled = torch.roll(perm, int(roll_offset), 0)
            active_idx = rolled[:k]
            sub_state[name] = param[active_idx].clone()
            active_indices[name] = active_idx

        pkg["sub_state"] = sub_state
        pkg["active_indices"] = active_indices
        pkg["capacity"] = capacity
        # Override regular_model_params: client trains on the sub-model
        pkg["regular_model_params"] = sub_state
        return pkg

    def aggregate_client_updates(self, packages: "OrderedDict[int, dict]") -> None:
        global_state = {name: p.clone() for name, p in self.public_model_params.items()}
        accum = {
            k: torch.zeros_like(v, dtype=torch.float64) for k, v in global_state.items()
        }
        count = {
            k: torch.zeros(v.shape[0] if v.dim() > 0 else 1, dtype=torch.float64)
            for k, v in global_state.items()
        }

        for cid, pkg in packages.items():
            sub_state = pkg["regular_model_params"]
            active_indices = pkg.get("active_indices", {})
            for name in global_state:
                if name not in sub_state or name not in active_indices:
                    continue
                idx = active_indices[name]
                accum[name][idx] += sub_state[name].to(dtype=torch.float64)
                count[name][idx] += 1

        new_global = OrderedDict()
        for name, orig in global_state.items():
            if orig.dim() == 0:
                new_global[name] = orig
                continue
            c = count[name]
            mask = c > 0
            if orig.dim() > 1:
                expand = mask.unsqueeze(-1).expand_as(orig)
            else:
                expand = mask
            new_global[name] = torch.where(
                expand,
                (accum[name] / c.clamp(min=1)).to(dtype=orig.dtype),
                orig,
            )

        self._commit_global(new_global)
        self.round_counter += 1


class FedRolex_Client(tFL_Client):
    """Client that trains a physically narrower sub-model extracted from the global model."""

    capacity: float = 1.0

    def set_parameters(self, package: dict) -> None:
        self.id = package["client_id"]
        self.current_iter = package["current_iter"]
        self._load_private(self.id)
        # Load optimizer and scheduler states (sub-model optimizer may mismatch shape)
        if package["optimizer_state"]:
            try:
                self.optimizer.load_state_dict(package["optimizer_state"])
            except Exception:
                self.optimizer.load_state_dict(self.init_optimizer_state)
        else:
            self.optimizer.load_state_dict(self.init_optimizer_state)
        if package["scheduler_state"]:
            try:
                self.scheduler.load_state_dict(package["scheduler_state"])
            except Exception:
                self.scheduler.load_state_dict(self.init_scheduler_state)
        else:
            self.scheduler.load_state_dict(self.init_scheduler_state)
        # Store sub-state for fit() — don't load into self.model (wrong shapes)
        self._sub_state = {k: v.clone() for k, v in package["regular_model_params"].items()}
        self._active_indices = package.get("active_indices", {})
        self.capacity = package.get("capacity", 1.0)

    def fit(self) -> None:
        self._set_worker_seed(self._loader_seed("train"))
        loader = self.load_train_data()

        if not self._sub_state:
            # Fallback: train full model normally
            super().fit()
            return

        sub_model = self._build_narrow_model()
        sub_model.to(self.device)
        sub_model.train()

        optimizer = torch.optim.Adam(sub_model.parameters(), lr=self.learning_rate)

        for _ in range(self.epochs):
            for batch_x, batch_y, x_mark, y_mark in loader:
                batch_x = batch_x.to(device=self.device, dtype=torch.float32, non_blocking=True)
                batch_y = batch_y.to(device=self.device, dtype=torch.float32, non_blocking=True)
                x_mark = x_mark.to(device=self.device, dtype=torch.float32, non_blocking=True)
                y_mark = y_mark.to(device=self.device, dtype=torch.float32, non_blocking=True)

                optimizer.zero_grad()
                output = sub_model(batch_x, x_mark=x_mark, y_mark=y_mark)
                target = batch_y
                if output.shape[1] != batch_y.shape[1] and self._active_indices:
                    last_key = list(self._active_indices.keys())[-1]
                    target = batch_y[:, self._active_indices[last_key], :]
                loss = self.loss(output, target)
                loss.backward()
                optimizer.step()

        self._sub_state = sub_model.state_dict()
        sub_model.to("cpu")

    def package(self, train_time: float) -> dict:
        return {
            "client_id": self.id,
            "regular_model_params": self._sub_state,
            "personal_model_params": {},
            "optimizer_state": {},
            "scheduler_state": copy.deepcopy(self.scheduler.state_dict()),
            "score": self.train_samples,
            "train_time": train_time,
            "active_indices": self._active_indices,
        }

    def _build_narrow_model(self):
        model = copy.deepcopy(self.model)

        new_state = model.state_dict()
        for name, sub_param in self._sub_state.items():
            if name not in new_state:
                continue
            orig_shape = new_state[name].shape
            sub_shape = sub_param.shape

            if sub_shape == orig_shape:
                new_state[name] = sub_param
                continue

            parts = name.split(".")
            module = model
            for p in parts[:-1]:
                module = getattr(module, p)
            attr_name = parts[-1]

            if attr_name == "weight" and len(sub_shape) == 2:
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
                parent = model
                for p in parts[:-2]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-2], new_linear)
            elif attr_name == "weight" and len(sub_shape) == 3:
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
                parent = model
                for p in parts[:-2]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-2], new_conv)
            else:
                new_state[name] = sub_param

        try:
            model.load_state_dict(new_state, strict=False)
        except RuntimeError:
            pass

        return model
