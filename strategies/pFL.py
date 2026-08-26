import copy
from argparse import Namespace
from collections import OrderedDict
from typing import Any, Dict, Optional

import numpy as np
import torch

from .base import SharedMethods
from .tFL import tFL, tFL_Client


class pFLShared:
    """Operations shared by personalized servers and stateless workers."""

    @staticmethod
    def personalized_model_state(
        base_state: "OrderedDict[str, torch.Tensor]",
        personal_params: Dict[str, Any],
        parameter_names: Optional[list[str]] = None,
    ) -> Optional[OrderedDict]:
        """Overlay a strategy's persisted personalized weights on the global model."""
        if isinstance(personal_params.get("model_per"), dict):
            personal_state = personal_params["model_per"]
        elif isinstance(personal_params.get("mask"), dict) and isinstance(
            personal_params.get("local_model_state"), dict
        ):
            mask = personal_params["mask"]
            local = personal_params["local_model_state"]
            personal_state = OrderedDict(
                (
                    name,
                    torch.where(
                        mask[name].bool(),
                        local[name].to(base_state[name]),
                        base_state[name],
                    ),
                )
                for name in mask
                if name in base_state and name in local
            )
        elif isinstance(personal_params.get("personalized_params"), (list, tuple)):
            personal_state = OrderedDict(
                zip(
                    parameter_names or base_state,
                    personal_params["personalized_params"],
                )
            )
        else:
            personal_state = OrderedDict(
                (name, value)
                for name, value in personal_params.items()
                if name in base_state and torch.is_tensor(value)
            )
        if not personal_state:
            return None
        state = OrderedDict(base_state)
        state.update(personal_state)
        return state


class pFL(pFLShared, tFL):
    """Personalized FL server."""

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        self._best_personal_loss: float = float("inf")
        m = self.metrics
        self.metrics = {
            "time_per_iter": m["time_per_iter"],
            "generalization_avg_train_loss": m["generalization_avg_train_loss"],
            "personalization_avg_train_loss": [],
            "generalization_avg_test_loss": m["generalization_avg_test_loss"],
            "personalization_avg_test_loss": [],
            "downlink_mb": m["downlink_mb"],
        }

    def package(self, client_id: int) -> dict[str, Any]:
        pkg = super().package(client_id=client_id)
        pkg["__wire__"] = ("regular_model_params", "personal_model_params")
        return pkg

    def _pre_eval_hook(self, dataset_type: str) -> None:
        incumbent = [i for i in range(self.num_clients) if not self.is_new[i]]
        losses = self.trainer.evaluate_personalized(
            ids=incumbent,
            global_params=self.public_model_params,
            personal_map=self.clients_personal_model_params,
            dataset_type=dataset_type,
            current_iter=self.current_iter,
        )
        metric = f"personalization_avg_{dataset_type}_loss"
        metric_val = float(np.mean(losses))
        self.metrics[metric].append(metric_val)
        self.logger.info(
            f"Personalization {dataset_type.capitalize()} Loss: "
            f"{self.metrics[metric][-1]:.4f}"
        )
        if dataset_type == "test":
            self._best_personal_loss = min(self._best_personal_loss, metric_val)
        for cid, loss in zip(incumbent, losses):
            self._round_client_data.setdefault(cid, {})[f"{dataset_type}_loss"] = float(
                loss
            )

    def _save_personal_models(self, postfix: str) -> None:
        tmp = copy.deepcopy(self.model)
        base_state = OrderedDict(
            (name, value.detach().cpu().clone())
            for name, value in self.model.state_dict().items()
        )
        for cid, params in self.clients_personal_model_params.items():
            state = self.personalized_model_state(
                base_state=base_state,
                personal_params=params,
                parameter_names=[name for name, _ in self.model.named_parameters()],
            )
            if state is None:
                continue
            tmp.load_state_dict(state, strict=False)
            SharedMethods.save_model(
                model=tmp,
                path=self.model_path,
                name=f"client_{cid}",
                postfix=postfix,
                configs=self.configs,
                verbose=self.logger,
            )

    def _save_best_hook(self) -> None:
        losses = [
            v
            for v in self.metrics.get("personalization_avg_test_loss", [])
            if v != self.default_value
        ]
        if not losses:
            super()._save_best_hook()
            return
        if losses[-1] != self._best_personal_loss:
            return
        SharedMethods.save_model(
            model=self.model,
            path=self.model_path,
            name=self.name.strip(),
            postfix="best",
            configs=self.configs,
            verbose=self.logger,
        )
        self._save_personal_models(postfix="best")

    def _save_last_hook(self) -> None:
        super()._save_last_hook()
        self._save_personal_models(postfix="last")

    def early_stopping(self) -> bool:
        metric = self.metrics["personalization_avg_test_loss"]
        if not self.patience or len(metric) < self.patience:
            return False
        if min(metric) not in metric[-self.patience :]:
            self.logger.info("Early stopping activated.")
            return True
        return False


class pFL_Client(pFLShared, tFL_Client):
    """Passthrough — same as tFL_Client; named subclass kept as the discovery anchor for ``<Strategy>_Client`` resolution and as the shared base for personalized-FL client classes."""

    def package(self) -> dict[str, Any]:
        pkg = super().package()
        if not self.return_diff:
            pkg["__wire__"] = ("regular_model_params", "personal_model_params")
        return pkg

    def evaluate_personalized(
        self,
        client_id: int,
        global_params: "OrderedDict[str, torch.Tensor]",
        personal_params: Dict[str, Any],
        dataset_type: str,
        current_iter: int,
    ) -> float:
        state = self.personalized_model_state(
            base_state=global_params,
            personal_params=personal_params,
            parameter_names=[name for name, _ in self.model.named_parameters()],
        )
        return super().evaluate_personalized(
            client_id=client_id,
            global_params=global_params,
            personal_params=state or {},
            dataset_type=dataset_type,
            current_iter=current_iter,
        )
