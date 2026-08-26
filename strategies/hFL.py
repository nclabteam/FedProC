import copy
import json
import os
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .pFL import pFL, pFL_Client
from .tFL import Trainer, tFL_Client


class hFLShared:
    """Utilities shared by heterogeneous-model strategies."""

    @staticmethod
    def load_public_data(
        configs: Namespace, dataset_name: str, batch_size: int
    ) -> DataLoader:
        import data_factory

        public_configs = copy.deepcopy(configs)
        public_configs.dataset = dataset_name
        dataset = getattr(data_factory, dataset_name)(public_configs)
        dataset.execute()

        arrays: dict[str, list[np.ndarray]] = {
            name: [] for name in ("x", "y", "x_mark", "y_mark")
        }
        for entry in dataset.info:
            with np.load(entry["paths"]["train"]) as data:
                for name in arrays:
                    arrays[name].append(data[name])
        tensors = [
            torch.as_tensor(np.concatenate(arrays[name]), dtype=torch.float32)
            for name in arrays
        ]
        return DataLoader(
            dataset=TensorDataset(*tensors),
            batch_size=batch_size,
            shuffle=True,
        )


class hFL_Trainer(Trainer):
    """Reuse one stateless worker per model prototype."""

    def __init__(
        self,
        server: "hFL",
        client_cls: type[tFL_Client],
        configs: Namespace,
        times: int,
    ) -> None:
        self.server = server
        self.client_cls = client_cls
        self.parallel = False
        self.workers: dict[int, tFL_Client] = {}
        self.client_prototypes: dict[int, int] = {}
        self.prototype_clients: dict[int, list[int]] = {}
        self.prototype_workers: dict[int, tFL_Client] = {}

        with open(server.path_info, encoding="utf-8") as stream:
            client_info = json.load(stream)
        prototype_ids: dict[tuple[str, str, str], int] = {}
        for entry in server.model_map:
            client_id = int(entry["client"])
            params = entry.get("params", {})
            dimensions = {
                name: client_info[client_id][name]
                for name in ("input_channels", "output_channels")
            }
            key = (
                entry["model"],
                json.dumps(params, sort_keys=True, default=str),
                json.dumps(dimensions, sort_keys=True, default=str),
            )
            if key not in prototype_ids:
                prototype_id = len(prototype_ids)
                prototype_ids[key] = prototype_id
                client_configs = copy.deepcopy(configs)
                client_configs.model = entry["model"]
                client_configs._worker_client_id = client_id
                for name, value in params.items():
                    setattr(client_configs, name, value)
                self.prototype_workers[prototype_id] = client_cls(
                    configs=client_configs,
                    times=times,
                    device=client_configs.device,
                )
                self.prototype_clients[prototype_id] = []
            prototype_id = prototype_ids[key]
            self.client_prototypes[client_id] = prototype_id
            self.prototype_clients[prototype_id].append(client_id)
            self.workers[client_id] = self.prototype_workers[prototype_id]

    def worker_for(self, client_id: int) -> tFL_Client:
        return self.workers[client_id]

    def train(self, selected: Sequence[int]) -> OrderedDict[int, dict[str, Any]]:
        packages: OrderedDict[int, dict[str, Any]] = OrderedDict()
        for client_id in selected:
            output = self._receive(
                cid=client_id,
                out=self.worker_for(client_id=client_id).train(
                    package=self._dispatch(cid=client_id)
                ),
            )
            self._write_back(cid=client_id, out=output)
            packages[client_id] = output
        return packages

    def evaluate(
        self,
        ids: list[int],
        global_params: Mapping[str, torch.Tensor],
        dataset_type: str,
        current_iter: int,
    ) -> list[float]:
        return [
            self.worker_for(client_id=client_id).evaluate_global(
                client_id=client_id,
                global_params=global_params,
                dataset_type=dataset_type,
                current_iter=current_iter,
            )
            for client_id in ids
        ]

    def evaluate_personalized(
        self,
        ids: list[int],
        global_params: Mapping[str, torch.Tensor],
        personal_map: Mapping[int, Mapping[str, Any]],
        dataset_type: str,
        current_iter: int,
    ) -> list[float]:
        return [
            self.worker_for(client_id=client_id).evaluate_personalized(
                client_id=client_id,
                global_params=global_params,
                personal_params=personal_map[client_id],
                dataset_type=dataset_type,
                current_iter=current_iter,
            )
            for client_id in ids
        ]


class hFL(hFLShared, pFL):
    """Base branch for client-specific model architectures."""

    optional = {
        "models": "DLinear,TSMixer",
        "model_config": "",
        "model_assign": "robin",
    }

    @classmethod
    def args_update(cls, parser: ArgumentParser) -> None:
        parser.add_argument("--models", type=str, default=None)
        parser.add_argument("--model_config", type=str, default=None)
        parser.add_argument(
            "--model_assign",
            type=str,
            default=None,
            choices=["robin", "wrap", "random"],
        )

    def __init__(self, configs: Namespace, times: int) -> None:
        self.set_configs(configs=configs, times=times)
        self.model_map = self._build_model_map(configs=configs)
        super().__init__(configs=configs, times=times)
        for client_id, worker in self.trainer.workers.items():
            self.clients_personal_model_params[client_id] = OrderedDict(
                (name, value.detach().cpu().clone())
                for name, value in worker.model.state_dict().items()
            )
        self._export_model_config()

    def _make_trainer(self) -> hFL_Trainer:
        return hFL_Trainer(
            server=self,
            client_cls=self._client_cls(),
            configs=self.configs,
            times=self.times,
        )

    @staticmethod
    def _parse_models_str(models_str: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for part in models_str.split(","):
            name, separator, count = part.strip().partition(":")
            if not name:
                raise ValueError("models cannot contain an empty name")
            result[name] = int(count) if separator else 1
            if result[name] <= 0:
                raise ValueError("model counts must be positive")
        return result

    def _build_model_map(self, configs: Namespace) -> list[dict[str, Any]]:
        if self.model_config:
            with open(self.model_config, encoding="utf-8") as stream:
                return json.load(stream)

        models = self._parse_models_str(models_str=self.models)
        model_list = [name for name, count in models.items() for _ in range(count)]
        if self.model_assign in {"robin", "wrap"}:
            assignments = [
                model_list[index % len(model_list)]
                for index in range(configs.num_clients)
            ]
        elif self.model_assign == "random":
            assignments = (
                np.random.default_rng(configs.seed)
                .choice(model_list, size=configs.num_clients)
                .tolist()
            )
        else:
            raise ValueError(f"unsupported model assignment: {self.model_assign}")

        result = []
        for client_id, model_name in enumerate(assignments):
            model_class = self._get_objective_function(
                func_type="models", func_name=model_name
            )
            result.append(
                {
                    "client": client_id,
                    "model": model_name,
                    "params": dict(getattr(model_class, "optional", {})),
                }
            )
        return result

    def _export_model_config(self) -> None:
        with open(
            os.path.join(self.save_path, "model_config.json"),
            "w",
            encoding="utf-8",
        ) as stream:
            json.dump(self.model_map, stream, indent=2)

    def package(self, client_id: int) -> dict[str, Any]:
        package = super().package(client_id=client_id)
        package["regular_model_params"] = copy.deepcopy(
            self.clients_personal_model_params[client_id]
        )
        package["personal_model_params"] = {}
        package["__wire__"] = ()
        return package

    def aggregate_client_updates(self, packages: Mapping[int, dict[str, Any]]) -> None:
        for client_id, package in packages.items():
            self.clients_personal_model_params[client_id].update(
                package["regular_model_params"]
            )

    def evaluate_generalization(self, dataset_type: str) -> None:
        """No shared model exists in the hFL branch."""

    def save_models(self, save_type: str) -> None:
        if save_type not in {"last", "best"}:
            raise ValueError("save_type must be 'last' or 'best'")
        if save_type == "best":
            losses = [
                value
                for value in self.metrics.get("personalization_avg_test_loss", [])
                if value != self.default_value
            ]
            if not losses or losses[-1] != min(losses):
                return

        for client_id, personal in self.clients_personal_model_params.items():
            if not personal:
                continue
            worker = self.trainer.worker_for(client_id=client_id)
            worker.model.load_state_dict(state_dict=personal, strict=False)
            self.save_model(
                model=worker.model,
                path=self.model_path,
                name=f"client_{client_id}_{worker.model.__class__.__name__}",
                postfix=save_type,
                configs=worker.configs,
                metadata={
                    "save_type": save_type,
                    "owner": f"client_{client_id}",
                },
                verbose=self.logger,
            )

    def _save_best_hook(self) -> None:
        self.save_models(save_type="best")

    def _save_last_hook(self) -> None:
        self.save_models(save_type="last")

    def get_model_info(self) -> None:
        if self.exclude_server_model_processes or not isinstance(
            getattr(self, "trainer", None), hFL_Trainer
        ):
            return
        for prototype_id, worker in self.trainer.prototype_workers.items():
            client_id = self.trainer.prototype_clients[prototype_id][0]
            worker._load_private(client_id=client_id)
            worker.id = client_id
            worker.current_iter = 0
            worker.name = f"client_{client_id}_{worker.model.__class__.__name__}"
            worker.models_info_path = self.models_info_path
            worker.summarize_model(dataloader=worker.load_train_data())


class hFL_Client(pFL_Client):
    """Stateless worker for one heterogeneous model prototype."""

    def __init__(self, configs: Namespace, times: int, device: str) -> None:
        super().__init__(configs=configs, times=times, device=device)
        self.regular_params_name = list(self.model.state_dict())

    def package(self) -> dict[str, Any]:
        package = super().package()
        package["__wire__"] = ()
        return package

    def evaluate_personalized(
        self,
        client_id: int,
        global_params: Mapping[str, torch.Tensor],
        personal_params: Mapping[str, torch.Tensor],
        dataset_type: str,
        current_iter: int,
    ) -> float:
        return tFL_Client.evaluate_personalized(
            self=self,
            client_id=client_id,
            global_params=OrderedDict(personal_params),
            personal_params={},
            dataset_type=dataset_type,
            current_iter=current_iter,
        )
