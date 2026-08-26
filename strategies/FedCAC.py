import copy
from typing import Any

import torch

from .pFL import pFL, pFL_Client


class FedCACShared:
    """Stateless FedCAC mask operations shared by server and client."""

    @staticmethod
    def critical_masks(previous: Any, current: Any, tau: float) -> Any:
        if not 0 <= tau <= 1:
            raise ValueError("FedCAC tau must be in [0, 1]")
        masks = []
        for before, after in zip(previous, current):
            scores = torch.abs((after.detach() - before.to(after.device)) * after)
            flat_mask = torch.zeros(scores.numel(), dtype=torch.int32)
            count = int(tau * scores.numel())
            if count:
                indices = torch.topk(
                    scores.reshape(-1), count, sorted=False
                ).indices.cpu()
                flat_mask[indices] = 1
            masks.append(flat_mask.reshape_as(scores))
        return masks

    @staticmethod
    def overlap_matrix(client_masks: Any) -> torch.Tensor:
        client_masks = [list(masks) for masks in client_masks]
        if not client_masks or any(not masks for masks in client_masks):
            raise ValueError("FedCAC masks must have the same nonzero size")
        flattened = [
            torch.cat([mask.reshape(-1) for mask in masks]).float()
            for masks in client_masks
        ]
        dimensions = flattened[0].numel()
        if not dimensions or any(mask.numel() != dimensions for mask in flattened):
            raise ValueError("FedCAC masks must have the same nonzero size")
        matrix = torch.stack(flattened)
        return 1.0 - torch.cdist(matrix, matrix, p=1) / (2.0 * dimensions)

    @classmethod
    def overlap_rate(cls, left_masks: Any, right_masks: Any) -> float:
        return float(cls.overlap_matrix(client_masks=[left_masks, right_masks])[0, 1])


class FedCAC(FedCACShared, pFL):
    """FedCAC: Federated Learning with Critical-parameter-Aware Collaboration (Wu et al., ICCV 2023)."""

    optional = {"tau": 0.5, "beta": 170}

    @classmethod
    def args_update(cls, parser: Any) -> None:
        parser.add_argument("--tau", type=float, default=None)
        parser.add_argument("--beta", type=int, default=None)

    def __init__(self, configs: Any, times: Any) -> None:
        super().__init__(configs=configs, times=times)
        mask_init = [
            torch.zeros(p.shape, dtype=torch.int32) for p in self.model.parameters()
        ]
        global_model_state = {
            k: v.cpu().clone() for k, v in self.model.state_dict().items()
        }
        for cid in range(self.num_clients):
            self.clients_personal_model_params[cid].update(
                {
                    "local_mask": [m.clone() for m in mask_init],
                    "customized_model_state": {
                        k: v.clone() for k, v in global_model_state.items()
                    },
                    "model_per": {k: v.clone() for k, v in global_model_state.items()},
                }
            )

    def select_clients(self) -> None:
        self._select_all_clients()

    def package(self, client_id: int) -> dict:
        package = super().package(client_id=client_id)
        package["customized_model_state"] = package["personal_model_params"][
            "customized_model_state"
        ]
        package["__wire__"] = ("regular_model_params", "customized_model_state")
        return package

    def aggregate_client_updates(self, packages: Any) -> None:
        cids = list(packages.keys())
        if not cids:
            raise ValueError("FedCAC requires at least one client")
        models = {cid: packages[cid]["regular_model_params"] for cid in cids}
        self._commit_global(new_params=self.mean_models(models=list(models.values())))
        for cid in cids:
            self.clients_personal_model_params[cid]["model_per"] = copy.deepcopy(
                models[cid]
            )

        if len(cids) == 1:
            self.clients_personal_model_params[cids[0]]["customized_model_state"] = (
                copy.deepcopy(models[cids[0]])
            )
            return

        overlaps = self.overlap_matrix(
            client_masks=[
                self.clients_personal_model_params[cid]["local_mask"] for cid in cids
            ]
        )
        off_diagonal = overlaps[~torch.eye(len(cids), dtype=torch.bool)]
        average = float(off_diagonal.mean())
        maximum = float(off_diagonal.max())
        threshold = average + (self.current_iter + 1) / self.beta * (maximum - average)

        for index, cid in enumerate(cids):
            rates = overlaps[index].tolist()
            collaborators = [cid] + [
                other
                for other_index, other in enumerate(cids)
                if other != cid and rates[other_index] >= threshold
            ]
            self.clients_personal_model_params[cid]["customized_model_state"] = (
                self.mean_models(models=[models[other] for other in collaborators])
            )


class FedCAC_Client(FedCACShared, pFL_Client):
    def set_parameters(self, package: dict) -> None:
        super().set_parameters(package=package)
        personal = package["personal_model_params"]
        combined = copy.deepcopy(package["regular_model_params"])
        customized = personal["customized_model_state"]
        for (name, _), mask in zip(
            self.model.named_parameters(), personal["local_mask"]
        ):
            mask = mask.to(combined[name].device, dtype=torch.bool)
            combined[name] = torch.where(mask, customized[name], combined[name])
        self.model.load_state_dict(combined, strict=False)

    def fit(self) -> None:
        prev_params = [p.detach().clone() for p in self.model.parameters()]
        super().fit()
        self._local_mask = self.critical_masks(
            previous=prev_params, current=list(self.model.parameters()), tau=self.tau
        )

    def package(self) -> dict:
        result = super().package()
        result["personal_model_params"]["local_mask"] = self._local_mask
        result["critical_mask"] = self._local_mask
        result["__wire__"] = ("regular_model_params", "critical_mask")
        return result
