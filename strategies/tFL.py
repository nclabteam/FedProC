"""Stateless-client / server-owned-state execution core."""

import copy
import csv
import json
import logging
import os
import time
from argparse import Namespace
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping
from contextlib import suppress
from typing import Any, Dict, List, Optional

import numpy as np
import ray
import torch
from torch.utils.data import DataLoader

from .base import SharedMethods

_PARITY_RNG_SHIM = True


class tFL_Client(SharedMethods):
    """Reusable worker that can *become* any client for a single round."""

    def __init__(self, configs: Namespace, times: int, device: str) -> None:
        self.set_configs(configs=configs, times=times)
        self.return_diff = bool(self.return_diff or type(self).return_diff)
        self.device = device
        self.id: Optional[int] = None
        self.current_iter = 0
        self.train_samples = 0
        self._private_cache: Dict[int, dict] = {}

        # Heterogeneous workers may need a representative client's dimensions.
        self._load_private(client_id=getattr(configs, "_worker_client_id", 0))
        self.initialize_model()
        self.optimizer = self._build(kind="optimizers", name=configs.optimizer)(
            params=self.model.parameters(), configs=configs
        )
        self._scheduler_base_lrs = [
            float(group["lr"]) for group in self.optimizer.param_groups
        ]
        self.initialize_scheduler()
        self.loss = self._build(kind="losses", name=configs.loss)()
        self.init_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        self.regular_params_name = [k for k, _ in self.model.named_parameters()]
        self.personal_params_name: List[str] = []

    @classmethod
    def _build(cls, kind: str, name: str) -> Callable[..., Any]:
        return cls._get_objective_function(func_type=kind, func_name=name)

    def _load_private(self, client_id: int) -> None:
        if client_id not in self._private_cache:
            with open(self.path_info, "r", encoding="utf-8") as f:
                self._private_cache[client_id] = json.load(f)[client_id]
        info = self._private_cache[client_id]
        self.train_file = info["paths"]["train"]
        self.test_file = info["paths"]["test"]
        self.stats = info["stats"]["train"]
        self.input_channels = info["input_channels"]
        self.output_channels = info["output_channels"]
        self.configs.__dict__["input_channels"] = self.input_channels
        self.configs.__dict__["output_channels"] = self.output_channels
        self.scaler = getattr(__import__("scalers"), self.configs.scaler)(self.stats)

    def _loader_seed(self, dataset_type: str) -> Optional[int]:
        if self.seed is None:
            return None
        offset = {"train": 0, "test": 1, "valid": 2}.get(dataset_type, 3)
        return self._derive_seed(
            int(self.seed) + int(self.times), self.id, self.current_iter, offset
        )

    def load_train_data(self) -> DataLoader:
        loader = self.load_data(
            file=self.train_file,
            sample_ratio=self.sample_ratio,
            shuffle=True,
            scaler=self.scaler,
            batch_size=self.batch_size,
            seed=self._loader_seed(dataset_type="train"),
        )
        self.train_samples = len(loader.dataset)
        return loader

    def load_test_data(self) -> DataLoader:
        return self.load_data(
            file=self.test_file,
            sample_ratio=1.0,
            shuffle=False,
            scaler=self.scaler,
            batch_size=self.batch_size,
            seed=self._loader_seed(dataset_type="test"),
        )

    return_diff: bool = False
    return_diff_score: bool = True

    def _warmup(self) -> None:
        self.model.to(self.device)
        b, s, c = 1, self.configs.input_len, self.configs.input_channels
        x = torch.zeros(b, s, c, dtype=torch.float32, device=self.device)
        try:
            self.model(x)
        except Exception:
            pass
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.model.to("cpu")

    def set_parameters(self, package: Dict[str, Any]) -> None:
        self.id = package["client_id"]
        self.current_iter = package["current_iter"]
        self._load_private(client_id=self.id)
        self.model.load_state_dict(package["regular_model_params"], strict=False)
        if package["personal_model_params"]:
            self.model.load_state_dict(package["personal_model_params"], strict=False)
        if package["optimizer_state"]:
            self.optimizer.load_state_dict(package["optimizer_state"])
        else:
            self.optimizer.load_state_dict(self.init_optimizer_state)
        if self.scheduler_mode == "iteration":
            self.restore_scheduler(
                scheduler=self.scheduler,
                optimizer=self.optimizer,
                state=package["scheduler_state"] or self.init_scheduler_state,
                mode=self.scheduler_mode,
            )
        if self.return_diff:
            state = self.model.state_dict()
            self._initial_regular_params = OrderedDict(
                (k, state[k].detach().cpu().clone()) for k in self.regular_params_name
            )

    def fit(self) -> None:
        self._set_worker_seed(seed=self._loader_seed(dataset_type="train"))
        loader = self.load_train_data()
        self.initialize_scheduler(steps_per_epoch=len(loader))
        offload_after_epoch = self.efficiency == "low"
        for _ in range(self.epochs):
            self.train_one_epoch(
                model=self.model,
                dataloader=loader,
                optimizer=self.optimizer,
                criterion=self.loss,
                scheduler=self.scheduler,
                device=self.device,
                offload_after=offload_after_epoch,
            )
        if self.efficiency == "med":
            self.model.to("cpu")

    def train(self, package: Dict[str, Any]) -> Dict[str, Any]:
        self.set_parameters(package=package)
        self.fit()
        return self.package()

    def package(self) -> Dict[str, Any]:
        state = self.model.state_dict()
        regular = {k: state[k].detach().cpu().clone() for k in self.regular_params_name}
        personal = {
            k: state[k].detach().cpu().clone() for k in self.personal_params_name
        }
        pkg = {
            "__wire__": ("regular_model_params", "score"),
            "client_id": self.id,
            "regular_model_params": regular,
            "personal_model_params": personal,
            "optimizer_state": self._optimizer_state_to_cpu(optimizer=self.optimizer),
            "scheduler_state": copy.deepcopy(self.scheduler.state_dict()),
            "score": self.train_samples,
        }
        if self.return_diff:
            pkg["model_params_diff"] = OrderedDict(
                (k, self._initial_regular_params[k] - regular[k])
                for k in self.regular_params_name
            )
            pkg["__wire__"] = (
                ("model_params_diff", "score")
                if self.return_diff_score
                else ("model_params_diff",)
            )
        return pkg

    def evaluate_global(
        self,
        client_id: int,
        global_params: "OrderedDict[str, torch.Tensor]",
        dataset_type: str,
        current_iter: int,
    ) -> float:
        self.id = client_id
        self.current_iter = current_iter
        self._load_private(client_id=client_id)
        self.model.load_state_dict(global_params, strict=False)
        loader = (
            self.load_test_data() if dataset_type == "test" else self.load_train_data()
        )
        losses = self.calculate_loss(
            model=self.model,
            dataloader=loader,
            criterion=self.loss,
            device=self.device,
            offload_after=self.efficiency != "high",
        )
        return float(np.mean(losses))

    def evaluate_personalized(
        self,
        client_id: int,
        global_params: "OrderedDict[str, torch.Tensor]",
        personal_params: Dict[str, torch.Tensor],
        dataset_type: str,
        current_iter: int,
    ) -> float:
        self.id = client_id
        self.current_iter = current_iter
        self._load_private(client_id=client_id)
        self.model.load_state_dict(global_params, strict=False)
        if personal_params:
            self.model.load_state_dict(personal_params, strict=False)
        loader = (
            self.load_test_data() if dataset_type == "test" else self.load_train_data()
        )
        losses = self.calculate_loss(
            model=self.model,
            dataloader=loader,
            criterion=self.loss,
            device=self.device,
            offload_after=self.efficiency != "high",
        )
        return float(np.mean(losses))


class Trainer:
    """Drives per-client work serially or across a Ray actor pool."""

    def __init__(
        self,
        server: "tFL",
        client_cls: type[tFL_Client],
        configs: Namespace,
        times: int,
    ) -> None:
        self.server = server
        self.client_cls = client_cls
        self.parallel = server.parallel
        if not self.parallel:
            self.worker = client_cls(
                configs=configs, times=times, device=configs.device
            )
        else:
            self.num_workers = int(server.num_workers)
            device = "cuda" if server.num_gpus > 0 else "cpu"
            remote_cls = ray.remote(num_gpus=server.num_gpus / self.num_workers)(
                client_cls
            )
            self.workers = [
                remote_cls.remote(configs=configs, times=times, device=device)
                for _ in range(self.num_workers)
            ]
            # Force each Ray actor to initialize (imports, CUDA context, model build)
            # before the first timed round starts.
            ray.get([w._warmup.remote() for w in self.workers])

    def _dispatch(self, cid: int) -> dict:
        pkg = self.server.package(client_id=cid)
        real_keys = pkg.pop("__wire__", ())
        self.server._downlink_sizes[cid] = sum(
            self.server.get_size(obj=pkg[k]) for k in real_keys if k in pkg
        )
        return pkg

    def _receive(self, cid: int, out: dict) -> dict:
        real_keys = out.pop("__wire__", ())
        self.server._uplink_sizes[cid] = sum(
            self.server.get_size(obj=out[k]) for k in real_keys if k in out
        )
        return out

    def train(self, selected: List[int]) -> "OrderedDict[int, dict]":
        packages: "OrderedDict[int, dict]" = OrderedDict()
        if not self.parallel:
            for cid in selected:
                out = self._receive(
                    cid=cid,
                    out=self.worker.train(package=self._dispatch(cid=cid)),
                )
                self._write_back(cid=cid, out=out)
                packages[cid] = out
            return packages

        idle = deque(range(self.num_workers))
        futures: list = []
        job_map: Dict[Any, tuple] = {}
        i = 0
        results: Dict[int, dict] = {}
        while i < len(selected) or futures:
            while i < len(selected) and idle:
                wid = idle.popleft()
                cid = selected[i]
                fut = self.workers[wid].train.remote(package=self._dispatch(cid=cid))
                job_map[fut] = (cid, wid)
                futures.append(fut)
                i += 1
            if futures:
                done, futures = ray.wait(futures)
                for fut in done:
                    cid, wid = job_map.pop(fut)
                    out = self._receive(cid=cid, out=ray.get(fut))
                    self._write_back(cid=cid, out=out)
                    results[cid] = out
                    idle.append(wid)
        return OrderedDict((cid, results[cid]) for cid in selected)

    def evaluate(
        self,
        ids: List[int],
        global_params: Mapping[str, torch.Tensor],
        dataset_type: str,
        current_iter: int,
    ) -> List[float]:
        if not self.parallel:
            return [
                self.worker.evaluate_global(
                    client_id=cid,
                    global_params=global_params,
                    dataset_type=dataset_type,
                    current_iter=current_iter,
                )
                for cid in ids
            ]
        gp = ray.put(global_params)
        futures = [
            self.workers[k % self.num_workers].evaluate_global.remote(
                client_id=cid,
                global_params=gp,
                dataset_type=dataset_type,
                current_iter=current_iter,
            )
            for k, cid in enumerate(ids)
        ]
        return list(ray.get(futures))

    def evaluate_personalized(
        self,
        ids: List[int],
        global_params: Mapping[str, torch.Tensor],
        personal_map: Mapping[int, Mapping[str, Any]],
        dataset_type: str,
        current_iter: int,
    ) -> List[float]:
        if not self.parallel:
            return [
                self.worker.evaluate_personalized(
                    client_id=cid,
                    global_params=global_params,
                    personal_params=personal_map[cid],
                    dataset_type=dataset_type,
                    current_iter=current_iter,
                )
                for cid in ids
            ]
        gp = ray.put(global_params)
        futures = [
            self.workers[k % self.num_workers].evaluate_personalized.remote(
                client_id=cid,
                global_params=gp,
                personal_params=personal_map[cid],
                dataset_type=dataset_type,
                current_iter=current_iter,
            )
            for k, cid in enumerate(ids)
        ]
        return list(ray.get(futures))

    def dispatch_one(self, cid: int, wid: int) -> Any:
        """Dispatch a single client to a specific Ray worker. Returns a future."""
        return self.workers[wid].train.remote(package=self._dispatch(cid=cid))

    def _write_back(self, cid: int, out: Dict[str, Any]) -> None:
        self.server.client_optimizer_states[cid] = out["optimizer_state"]
        self.server.client_scheduler_states[cid] = out["scheduler_state"]
        self.server.clients_personal_model_params[cid].update(
            out["personal_model_params"]
        )


class tFL(SharedMethods):
    """Server that owns all per-client state and aggregates a global model."""

    # Class-level sentinels for optional metrics that may never be set
    # (None-stripping means None-default optionals never reach the instance).
    new_client_gen_test_loss: Optional[float] = None
    new_client_pers_test_loss: Optional[float] = None

    # Package keys that count as real network payload (exclude optimizer/scheduler state).
    def __init__(self, configs: Namespace, times: int) -> None:
        self.set_configs(configs=configs, times=times)
        self.mkdir()
        self.current_iter = 0
        self.num_join_clients = max(1, int(self.num_clients * self.join_ratio))
        self.current_num_join_clients = self.num_join_clients

        device_ids = [d for d in self.device_id.split(",") if d]
        self.num_gpus = len(device_ids) if self.device == "cuda" else 0
        self.parallel = self.num_gpus > 0 and self.num_workers > 0
        ray.init(
            num_gpus=self.num_gpus,
            ignore_reinit_error=True,
            logging_level=logging.ERROR,
            log_to_driver=False,
        )

        self.name = "  SERVER  "
        self.metrics = {
            "time_per_iter": [],
            "generalization_avg_train_loss": [],
            "generalization_avg_test_loss": [],
            "downlink_mb": [],
        }
        self._best_global_loss: float = float("inf")
        self._round_client_data: Dict[int, Dict[str, float]] = {}
        self._downlink_sizes: Dict[int, float] = {}
        self._uplink_sizes: Dict[int, float] = {}
        self.make_logger(name=self.name, path=self.log_path)

        with open(self.path_info, "r", encoding="utf-8") as f:
            info0 = json.load(f)[0]
        self.configs.__dict__["input_channels"] = info0["input_channels"]
        self.input_channels = info0["input_channels"]
        self.configs.__dict__["output_channels"] = info0["output_channels"]
        self.output_channels = info0["output_channels"]

        model_cls = self._get_objective_function(
            func_type="models", func_name=self.model
        )
        if _PARITY_RNG_SHIM:
            for _ in range(self.num_clients):
                model_cls(configs=self.configs)
        self.initialize_model()
        self.public_model_params = OrderedDict(
            (k, v.detach().cpu().clone()) for k, v in self.model.named_parameters()
        )

        self.client_optimizer_states = {i: {} for i in range(self.num_clients)}
        self.client_scheduler_states = {i: {} for i in range(self.num_clients)}
        self.clients_personal_model_params = {i: {} for i in range(self.num_clients)}
        self.is_new = {i: False for i in range(self.num_clients)}
        if self.exclude_ratio > 0.0:
            num_new = max(1, int(self.num_clients * self.exclude_ratio))
            rng = np.random.default_rng(self.seed)
            new_ids = set(rng.choice(self.num_clients, num_new, replace=False).tolist())
            for cid in new_ids:
                self.is_new[cid] = True
            self.logger.info(f"New clients ({num_new}): {sorted(new_ids)}")

        self.trainer = self._make_trainer()
        self.get_model_info()

    def _make_trainer(self) -> Trainer:
        return Trainer(
            server=self,
            client_cls=self._client_cls(),
            configs=self.configs,
            times=self.times,
        )

    def get_model_info(self) -> None:
        if self.exclude_server_model_processes:
            return
        if not self.parallel:
            worker = self.trainer.worker
        else:
            worker = self._client_cls()(
                configs=self.configs, times=self.times, device=self.device
            )
        worker._load_private(client_id=0)
        worker.id = 0
        worker.current_iter = 0
        dl = worker.load_train_data()
        self.summarize_model(dataloader=dl)

    def _client_cls(self) -> type[tFL_Client]:
        module_name = self.__module__
        class_name = self.__class__.__name__ + "_Client"
        try:
            return getattr(__import__(module_name, fromlist=[class_name]), class_name)
        except (ImportError, AttributeError):
            return tFL_Client

    def select_clients(self) -> None:
        incumbent = [i for i in range(self.num_clients) if not self.is_new[i]]
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(
                range(self.num_join_clients, len(incumbent) + 1), 1, replace=False
            )[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        self.selected_clients = [
            int(c)
            for c in np.random.choice(
                incumbent, self.current_num_join_clients, replace=False
            )
        ]

    def _select_all_clients(self) -> None:
        self.selected_clients = [
            i for i in range(self.num_clients) if not self.is_new[i]
        ]
        self.current_num_join_clients = len(self.selected_clients)

    def _select_one_client(self) -> None:
        incumbent = [i for i in range(self.num_clients) if not self.is_new[i]]
        if not incumbent:
            raise ValueError("at least one incumbent client is required")
        self.selected_clients = [int(np.random.choice(incumbent))]
        self.current_num_join_clients = 1

    def package(self, client_id: int) -> Dict[str, Any]:
        return {
            "__wire__": ("regular_model_params",),
            "client_id": client_id,
            "current_iter": self.current_iter,
            "regular_model_params": copy.deepcopy(self.public_model_params),
            "personal_model_params": self.clients_personal_model_params[client_id],
            "optimizer_state": self.client_optimizer_states[client_id],
            "scheduler_state": self.client_scheduler_states[client_id],
        }

    def _commit_global(self, new_params: Mapping[str, torch.Tensor]) -> None:
        self.public_model_params = OrderedDict(new_params)
        self.model.load_state_dict(self.public_model_params, strict=False)

    def _downlink_payload(self) -> Dict[str, Any]:
        return {}

    def _compute_send_mb(
        self, packages: Mapping[int, dict[str, Any]]
    ) -> tuple[dict[int, float], float]:
        uplink = {cid: self._uplink_sizes.get(cid, 0.0) for cid in packages}
        post_agg = self._downlink_payload()
        if post_agg:
            downlink = sum(
                self.get_size(obj=value) for value in post_agg.values()
            ) * len(self.selected_clients)
        else:
            downlink = sum(
                self._downlink_sizes.get(cid, 0.0) for cid in self.selected_clients
            )
        return uplink, downlink

    def train_one_round(self) -> dict:
        packages = self.trainer.train(selected=self.selected_clients)
        self.aggregate_client_updates(packages=packages)
        return packages

    @staticmethod
    def extract_models_and_scores(
        packages: Mapping[int, Mapping[str, Any]],
        model_key: str = "regular_model_params",
    ) -> tuple[List[Dict[str, torch.Tensor]], List[float]]:
        """Extract model payloads and scores in one pass."""
        models: List[Dict[str, torch.Tensor]] = []
        scores: List[float] = []
        for package in packages.values():
            models.append(package[model_key])
            scores.append(float(package["score"]))
        return models, scores

    def aggregate_client_updates(self, packages: "OrderedDict[int, dict]") -> None:
        models, scores = self.extract_models_and_scores(packages=packages)
        self._commit_global(
            new_params=self.mean_models(
                models=models,
                weights=scores,
            )
        )

    def _pre_eval_hook(self, dataset_type: str) -> None:
        """No-op for tFL; pFL overrides to run per-client personalized eval."""

    def evaluate_generalization(self, dataset_type: str) -> None:
        incumbent = [i for i in range(self.num_clients) if not self.is_new[i]]
        losses = self.trainer.evaluate(
            ids=incumbent,
            global_params=self.public_model_params,
            dataset_type=dataset_type,
            current_iter=self.current_iter,
        )
        metric = f"generalization_avg_{dataset_type}_loss"
        metric_val = float(np.mean(losses))
        self.metrics[metric].append(metric_val)
        self.logger.info(
            f"Generalization {dataset_type.capitalize()} Loss: "
            f"{self.metrics[metric][-1]:.4f}"
        )
        if dataset_type == "test":
            self._best_global_loss = min(self._best_global_loss, metric_val)
        for cid, loss in zip(incumbent, losses):
            self._round_client_data.setdefault(cid, {})[f"{dataset_type}_loss"] = float(
                loss
            )

    def early_stopping(self) -> bool:
        metric = self.metrics["generalization_avg_test_loss"]
        if not self.patience or len(metric) < self.patience:
            return False
        if min(metric) not in metric[-self.patience :]:
            self.logger.info("Early stopping activated.")
            return True
        return False

    def _flush_server_metrics(self) -> None:
        path = os.path.join(self.result_path, self.name.lower().strip() + ".csv")
        row = {"round": self.current_iter}
        for k, v in self.metrics.items():
            row[k] = v[-1] if v else ""
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    _client_csv_fields: tuple = ("uplink_mb", "train_loss", "test_loss")

    def _flush_client_data(self) -> None:
        fields = self._client_csv_fields
        for cid, data in self._round_client_data.items():
            path = os.path.join(self.result_path, f"client_{cid}.csv")
            row = {"round": self.current_iter}
            for f in fields:
                row[f] = data.get(f, "")
            write_header = not os.path.exists(path)
            with open(path, "a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        self._round_client_data.clear()

    def _flush_round(self) -> None:
        self._flush_server_metrics()
        self._flush_client_data()

    def _save_best_hook(self) -> None:
        vals = self.metrics.get("generalization_avg_test_loss", [])
        if not vals:
            return
        if vals[-1] == self._best_global_loss:
            self.save_model(
                model=self.model,
                path=self.model_path,
                name=self.name.strip(),
                postfix="best",
                configs=self.configs,
                verbose=self.logger,
            )

    def _save_last_hook(self) -> None:
        self.save_model(
            model=self.model,
            path=self.model_path,
            name=self.name.strip(),
            postfix="last",
            configs=self.configs,
            verbose=self.logger,
        )
        path = os.path.join(self.result_path, self.name.lower().strip() + ".csv")
        self.logger.info(f"Results saved to {path}")
        self.logger.info(f"Per-client results saved to {self.result_path}")

    def _finish_training(self) -> None:
        """Save final state and close the training runtime."""
        self._save_last_hook()
        with suppress(Exception):
            self.close_logger()
        with suppress(Exception):
            ray.shutdown()

    def train(self) -> None:
        for i in range(self.iterations):
            round_start = time.time()
            self.current_iter = i
            self.logger.info("")
            self.logger.info(
                f"-------------Round number: {str(i).zfill(4)}-------------"
            )
            self.select_clients()
            if i % self.eval_gap == 0:
                for dataset_type in ["train", "test"]:
                    if dataset_type == "train" and self.skip_eval_train:
                        continue
                    self._pre_eval_hook(dataset_type=dataset_type)
            packages = self.train_one_round()
            uplink, downlink = self._compute_send_mb(packages=packages)
            self.metrics["downlink_mb"].append(downlink)
            for cid, mb in uplink.items():
                self._round_client_data.setdefault(cid, {})["uplink_mb"] = mb
            if i % self.eval_gap == 0:
                for dataset_type in ["train", "test"]:
                    if dataset_type == "train" and self.skip_eval_train:
                        continue
                    if not self.exclude_server_model_processes:
                        self.evaluate_generalization(dataset_type=dataset_type)
                self._save_best_hook()
            iter_time = time.time() - round_start
            self.metrics["time_per_iter"].append(iter_time)
            self.logger.info(f"{iter_time:.2f}s")
            self._flush_round()
            if self.early_stopping():
                break
        self._finish_training()
