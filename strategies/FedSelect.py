import copy
import time
from argparse import Namespace
from typing import Any, Dict, List

import torch

from .pFL import pFL, pFL_Client


class FedSelect(pFL):
    """
    FedSelect: Personalized Federated Learning with Customized Selection of
    Parameters for Fine-Tuning.

    Each client maintains a binary mask per parameter element (0=global,
    1=local).  Initially all parameters are global.  Every `delta_interval`
    rounds the client's training delta |w_after - w_before| is computed for
    each global element; the top `prune_percent` fraction (largest delta) are
    promoted to local, up to a maximum sparsity of `sparsity_bound`.

    Aggregation is per-element: for each position only clients whose mask is 0
    (global) contribute.  Clients receive the server's global values only for
    their mask-0 positions; local positions are never overwritten.

    Reference: Tamirisa et al., "FedSelect: Personalized Federated Learning
    with Customized Selection of Parameters for Fine-Tuning", CVPR 2024.
    arXiv 2404.02478.
    """

    optional = {
        "prune_percent": 0.1,
        "delta_interval": 1,
        "sparsity_bound": 0.5,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--prune_percent", type=float, default=None)
        parser.add_argument("--delta_interval", type=int, default=None)
        parser.add_argument("--sparsity_bound", type=float, default=None)

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        self.parallel = False
        # {client_id: {param_name: float_tensor (0=global, 1=local)}}
        self._client_masks: Dict[int, Dict[str, torch.Tensor]] = {}
        # Snapshot of server params sent to clients this round (for delta)
        self._sent_params: Dict[str, torch.Tensor] = {}
        self._round = 0

    def _ensure_init(self) -> None:
        if self._client_masks:
            return
        for client in self.clients:
            self._client_masks[client.id] = {
                name: torch.zeros_like(p.data, dtype=torch.float32, device="cpu")
                for name, p in self.model.named_parameters()
            }

    def variables_to_be_sent(self) -> Dict[str, Any]:
        self._ensure_init()
        # Snapshot global params before sending (used for delta later)
        self._sent_params = {
            name: p.data.clone().cpu()
            for name, p in self.model.named_parameters()
        }
        masks = [self._client_masks[c.id] for c in self.clients]
        return {"model": self.model, "mask": masks}

    def receive_from_clients(self) -> None:
        self.client_data = []
        for client in self.selected_clients:
            data = client.send_to_server()
            self.client_data.append(data)

        # Update masks if it's a delta_interval round
        self._round += 1
        if self._round % self.delta_interval == 0:
            for client, cd in zip(self.selected_clients, self.client_data):
                self._update_mask(client.id, cd["model"])

    def _update_mask(self, client_id: int, trained_model) -> None:
        mask = self._client_masks[client_id]
        # Compute sparsity: fraction of elements already local
        total_elem = sum(m.numel() for m in mask.values())
        local_elem = sum(m.sum().item() for m in mask.values())
        if total_elem == 0 or (local_elem / total_elem) >= self.sparsity_bound:
            return

        # Collect deltas for currently-global elements
        global_deltas: List[torch.Tensor] = []
        for name, param in trained_model.named_parameters():
            if name not in mask:
                continue
            global_mask = (mask[name] == 0)
            if not global_mask.any():
                continue
            delta = (param.data.cpu() - self._sent_params[name]).abs()
            global_deltas.append(delta[global_mask].flatten())

        if not global_deltas:
            return

        all_deltas = torch.cat(global_deltas)
        # Threshold: top prune_percent of global elements by delta → local
        k = max(1, int(self.prune_percent * all_deltas.numel()))
        threshold = all_deltas.topk(k).values.min()

        # Mark elements with delta >= threshold as local (cap at sparsity_bound)
        remaining_budget = int(self.sparsity_bound * total_elem - local_elem)
        promoted = 0
        for name, param in trained_model.named_parameters():
            if name not in mask or promoted >= remaining_budget:
                break
            global_mask = (mask[name] == 0)
            delta = (param.data.cpu() - self._sent_params[name]).abs()
            to_promote = global_mask & (delta >= threshold)
            can_promote = min(to_promote.sum().item(), remaining_budget - promoted)
            if can_promote <= 0:
                continue
            # Only promote 'can_promote' elements (highest delta first)
            if to_promote.sum().item() > can_promote:
                flat_delta = delta.flatten()
                flat_mask = global_mask.flatten()
                eligible = flat_delta * flat_mask.float()
                top_positions = eligible.topk(can_promote).indices
                flat_new = mask[name].flatten().clone()
                flat_new[top_positions] = 1.0
                self._client_masks[client_id][name] = flat_new.view(mask[name].shape)
            else:
                self._client_masks[client_id][name] = torch.where(
                    to_promote, torch.ones_like(mask[name]), mask[name]
                )
            promoted += can_promote

    def aggregate_models(self) -> None:
        # Per-element weighted average only over clients where mask=0 (global)
        param_names = [name for name, _ in self.model.named_parameters()]
        total_score = sum(cd["score"] for cd in self.client_data)

        # Accumulators
        sum_vals: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(p.data, device="cpu")
            for name, p in self.model.named_parameters()
        }
        sum_counts: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(p.data, dtype=torch.float32, device="cpu")
            for name, p in self.model.named_parameters()
        }

        for client, cd in zip(self.selected_clients, self.client_data):
            weight = cd["score"] / total_score
            cid = client.id
            mask = self._client_masks[cid]
            for name, param in cd["model"].named_parameters():
                if name not in sum_vals:
                    continue
                global_w = (1.0 - mask[name])  # 1 where global, 0 where local
                sum_vals[name].add_(param.data.cpu() * global_w, alpha=weight)
                sum_counts[name].add_(global_w * weight)

        # Update server model: only positions with contributions
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                count = sum_counts[name]
                has_contrib = count > 0
                new_val = torch.where(
                    has_contrib,
                    sum_vals[name] / count.clamp(min=1e-8),
                    param.data.cpu(),
                )
                param.data.copy_(new_val)

        # Sync each client.model for evaluation (apply mask merge)
        client_by_id = {c.id: c for c in self.clients}
        for cd, client in zip(self.client_data, self.selected_clients):
            cid = client.id
            mask = self._client_masks[cid]
            if cid not in client_by_id:
                continue
            cli = client_by_id[cid]
            with torch.no_grad():
                for (name, cli_param), (_, srv_param) in zip(
                    cli.model.named_parameters(),
                    self.model.named_parameters(),
                ):
                    if name not in mask:
                        continue
                    global_w = (mask[name] == 0).float()
                    merged = global_w * srv_param.data.cpu() + (1 - global_w) * cli_param.data.cpu()
                    cli_param.data.copy_(merged)

    def save_models(self, save_type: str) -> None:
        if save_type not in ["last", "best"]:
            raise ValueError
        should_save = True
        if save_type == "best":
            vals = self.metrics.get("personal_avg_test_loss", [])
            if not vals or vals[-1] != min(vals):
                should_save = False
        if not should_save:
            return
        if not self.exclude_server_model_processes:
            self.save_model(
                model=self.model, path=self.model_path, name=self.name,
                postfix=save_type, configs=self.configs,
                metadata={"save_type": save_type, "owner": "server"},
                verbose=self.logger,
            )
        for client in self.clients:
            client.save_model(
                model=client.model, path=client.model_path, name=client.name,
                postfix=save_type, configs=client.configs,
                metadata={"save_type": save_type, "owner": client.name},
                verbose=client.logger,
            )


class FedSelect_Client(pFL_Client):
    """
    Client for FedSelect. On receive, merges server global params into local
    model for mask=0 positions. Trains normally; server computes delta.
    """

    def __init__(self, configs: Namespace, id: int, times: int) -> None:
        super().__init__(configs=configs, id=id, times=times)
        self._mask: Dict[str, torch.Tensor] = {}

    def receive_from_server(self, data: dict) -> None:
        self._mask = data["mask"]
        # Merge: global positions (mask=0) get server values; local stay
        with torch.no_grad():
            for (name, local_param), (_, srv_param) in zip(
                self.model.named_parameters(),
                data["model"].named_parameters(),
            ):
                if name not in self._mask:
                    local_param.data.copy_(srv_param.data)
                    continue
                global_w = (self._mask[name] == 0).to(local_param.device).float()
                merged = global_w * srv_param.data.to(local_param.device) + \
                         (1.0 - global_w) * local_param.data
                local_param.data.copy_(merged)

    def train(self):
        train_loader = self.load_train_data()
        start_time = time.time()

        self.model.to(self.device)
        self.model.train()
        for _ in range(self.epochs):
            for batch_x, batch_y, x_mark, y_mark in train_loader:
                self.optimizer.zero_grad()
                batch_x = batch_x.to(device=self.device, dtype=torch.float32)
                batch_y = batch_y.to(device=self.device, dtype=torch.float32)
                x_mark = x_mark.to(device=self.device, dtype=torch.float32)
                y_mark = y_mark.to(device=self.device, dtype=torch.float32)
                outputs = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
                loss = self.loss(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

        if self.efficiency != "high":
            self.model.to("cpu")
        self.metrics["train_time"].append(time.time() - start_time)

    def variables_to_be_sent(self) -> Dict[str, Any]:
        return {"model": self.model, "score": self.train_samples}
