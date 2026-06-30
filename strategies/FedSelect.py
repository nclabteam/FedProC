import copy
from argparse import Namespace
from collections import OrderedDict
from typing import Any, Dict, List

import numpy as np
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
    their mask-0 positions; local positions are carried forward from the
    previous round.

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
        self._round = 0
        self._pending_sent_params: Dict[int, Dict[str, torch.Tensor]] = {}
        self._pending_masks: Dict[int, Dict[str, torch.Tensor]] = {}
        init_mask = {
            name: torch.zeros_like(p.cpu(), dtype=torch.float32)
            for name, p in self.public_model_params.items()
        }
        init_local_state = {name: p.cpu().clone() for name, p in self.model.named_parameters()}
        for cid in range(self.num_clients):
            self.clients_personal_model_params[cid]["mask"] = {
                name: t.clone() for name, t in init_mask.items()
            }
            self.clients_personal_model_params[cid]["local_model_state"] = {
                name: t.clone() for name, t in init_local_state.items()
            }

    def package(self, client_id: int) -> dict:
        pkg = super().package(client_id)
        pm = self.clients_personal_model_params[client_id]
        current_mask = copy.deepcopy(pm["mask"])
        self._pending_masks[client_id] = current_mask
        self._pending_sent_params[client_id] = {
            name: p.clone().cpu() for name, p in self.public_model_params.items()
        }
        pkg["mask"] = current_mask
        return pkg

    def aggregate_client_updates(self, packages: "OrderedDict[int, dict]") -> None:
        self._round += 1

        # Update masks if delta interval
        if self._round % self.delta_interval == 0:
            for cid, pkg in packages.items():
                self._update_mask(cid, pkg["regular_model_params"], self._pending_sent_params[cid])

        scores = [p["score"] for p in packages.values()]
        total = float(sum(scores))
        cids = list(packages.keys())

        new_global = OrderedDict()
        for name, server_p in self.public_model_params.items():
            sum_val = torch.zeros_like(server_p.float())
            sum_count = torch.zeros_like(server_p.float())
            for cid, pkg in packages.items():
                weight = packages[cid]["score"] / total
                mask = self._pending_masks[cid][name]
                global_w = 1.0 - mask
                trained = pkg["regular_model_params"][name].float()
                sum_val.add_(trained * global_w, alpha=weight)
                sum_count.add_(global_w * weight)

            has_contrib = sum_count > 0
            new_global[name] = torch.where(
                has_contrib,
                sum_val / sum_count.clamp(min=1e-8),
                server_p.float(),
            ).to(server_p.dtype)

        self._commit_global(new_global)

    def _update_mask(
        self,
        client_id: int,
        trained_state: Dict[str, torch.Tensor],
        sent_params: Dict[str, torch.Tensor],
    ) -> None:
        mask = self.clients_personal_model_params[client_id]["mask"]
        total_elem = sum(m.numel() for m in mask.values())
        local_elem = sum(m.sum().item() for m in mask.values())
        if total_elem == 0 or (local_elem / total_elem) >= self.sparsity_bound:
            return

        # Collect deltas for currently-global elements
        global_deltas: List[torch.Tensor] = []
        for name, trained_p in trained_state.items():
            if name not in mask or name not in sent_params:
                continue
            global_mask = mask[name] == 0
            if not global_mask.any():
                continue
            delta = (trained_p.float() - sent_params[name].float()).abs()
            global_deltas.append(delta[global_mask].flatten())

        if not global_deltas:
            return

        all_deltas = torch.cat(global_deltas)
        k = max(1, int(self.prune_percent * all_deltas.numel()))
        threshold = all_deltas.topk(k).values.min()

        remaining_budget = int(self.sparsity_bound * total_elem - local_elem)
        promoted = 0
        for name, trained_p in trained_state.items():
            if name not in mask or promoted >= remaining_budget:
                break
            global_m = mask[name] == 0
            delta = (trained_p.float() - sent_params[name].float()).abs()
            to_promote = global_m & (delta >= threshold)
            can_promote = min(int(to_promote.sum().item()), remaining_budget - promoted)
            if can_promote <= 0:
                continue
            if int(to_promote.sum().item()) > can_promote:
                flat_delta = delta.flatten()
                flat_mask = global_m.flatten()
                eligible = flat_delta * flat_mask.float()
                top_positions = eligible.topk(can_promote).indices
                flat_new = mask[name].flatten().clone()
                flat_new[top_positions] = 1.0
                self.clients_personal_model_params[client_id]["mask"][name] = (
                    flat_new.view(mask[name].shape)
                )
            else:
                self.clients_personal_model_params[client_id]["mask"][name] = (
                    torch.where(to_promote, torch.ones_like(mask[name]), mask[name])
                )
            promoted += can_promote


class FedSelect_Client(pFL_Client):
    """
    Client for FedSelect. Receives the server's global model plus a per-element
    mask; mask=1 positions are overwritten with the client's previous local params
    before training.  Sends back the fully trained model; the server handles
    the per-element global aggregation and mask updates.
    """

    def set_parameters(self, package: dict) -> None:
        super().set_parameters(package)
        pm = package["personal_model_params"]
        self._mask = pm["mask"]
        local_state = pm["local_model_state"]
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name not in self._mask:
                    continue
                lw = self._mask[name].to(param.device)
                param.data = (
                    (1 - lw) * param.data + lw * local_state[name].to(param.device)
                )

    def package(self) -> dict:
        out = super().package()
        out["personal_model_params"]["local_model_state"] = {
            name: p.detach().cpu().clone()
            for name, p in self.model.named_parameters()
        }
        return out

    def evaluate_personalized(
        self,
        client_id: int,
        global_params: "OrderedDict[str, torch.Tensor]",
        personal_params: Dict[str, Any],
        dataset_type: str,
        current_iter: int,
    ) -> float:
        self.id = client_id
        self.current_iter = current_iter
        self._load_private(client_id)
        self.model.load_state_dict(global_params, strict=False)
        mask = personal_params["mask"]
        local_state = personal_params["local_model_state"]
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name not in mask:
                    continue
                lw = mask[name].to(param.device)
                param.data = (
                    (1 - lw) * param.data + lw * local_state[name].to(param.device)
                )
        loader = (
            self.load_test_data()
            if dataset_type == "test"
            else self.load_train_data()
        )
        losses = self.calculate_loss(
            model=self.model,
            dataloader=loader,
            criterion=self.loss,
            device=self.device,
            offload_after=self.efficiency != "high",
        )
        return float(np.mean(losses))
