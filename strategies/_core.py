"""Stateless-client / server-owned-state execution core (v1.12 prototype).

This is the FL-bench-style alternative to the stateful-client model in
``tFL``/``tFL_Client``.  Clients become reusable, stateless workers; ALL
per-client persistent state (model params, optimizer/scheduler state, personal
params) lives on the server and is threaded in/out each round via packages.

Status: prototype proving FedAvg parity against the legacy core.  Only the
serial path is exercised by the parity harness; the parallel actor-pool path
mirrors it for when GPUs are available.

A note on the ``_PARITY_RNG_SHIM`` below: the legacy core builds one model per
client (``initialize_clients``) before building the server's global model, so
the global model's initial weights sit at a specific torch-RNG offset.  To prove
the *training/aggregation/eval* path is numerically identical, we reproduce that
offset by building ``num_clients`` throwaway models before the global one.  This
shim is a validation aid only and is expected to be removed (with goldens
re-baselined) once the path is trusted.
"""

import copy
import json
import logging
import time
from argparse import Namespace
from collections import OrderedDict, deque
from typing import Any, Dict, List, Optional

import numpy as np
import ray
import torch

from .base import SharedMethods

_PARITY_RNG_SHIM = True


class StatelessClient(SharedMethods):
    """Reusable worker that can *become* any client for a single round.

    Holds one model/optimizer/scheduler/loss, rebuilt only at construction.
    Every round, :meth:`set_parameters` fully (re)loads the target client's
    data, model params, and optimizer/scheduler state from the server package,
    so no cross-round state lives on this object.
    """

    def __init__(self, configs: Namespace, times: int, device: str) -> None:
        # Copy every config onto the worker so strategy-specific hyper-parameters
        # (mu, num_global_layers, ...) are available, then override the device.
        self.set_configs(configs=configs, times=times)
        self.device = device
        self.id: Optional[int] = None
        self.current_iter = 0
        self.train_samples = 0
        self._private_cache: Dict[int, dict] = {}

        self._load_private(0)  # set channels on configs before building the model
        self.model = self._build("models", configs.model)(configs=configs)
        self.optimizer = self._build("optimizers", configs.optimizer)(
            params=self.model.parameters(), configs=configs
        )
        self.scheduler = self._build("schedulers", configs.scheduler)(
            optimizer=self.optimizer, configs=configs
        )
        self.loss = self._build("losses", configs.loss)()
        self.init_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        self.init_scheduler_state = copy.deepcopy(self.scheduler.state_dict())
        self.regular_params_name = [k for k, _ in self.model.named_parameters()]
        self.personal_params_name: List[str] = []

    # ── construction helpers ────────────────────────────────────────────────
    @staticmethod
    def _build(kind: str, name: str):
        return SharedMethods._get_objective_function(kind, name)

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

    # ── data ────────────────────────────────────────────────────────────────
    def _loader_seed(self, dataset_type: str) -> Optional[int]:
        if self.seed is None:
            return None
        offset = {"train": 0, "test": 1, "valid": 2}.get(dataset_type, 3)
        return self._derive_seed(
            int(self.seed) + int(self.times), self.id, self.current_iter, offset
        )

    def load_train_data(self):
        loader = self.load_data(
            file=self.train_file,
            sample_ratio=self.sample_ratio,
            shuffle=True,
            scaler=self.scaler,
            batch_size=self.batch_size,
            seed=self._loader_seed("train"),
        )
        self.train_samples = len(loader.dataset)
        return loader

    def load_test_data(self):
        return self.load_data(
            file=self.test_file,
            sample_ratio=1.0,
            shuffle=False,
            scaler=self.scaler,
            batch_size=self.batch_size,
            seed=self._loader_seed("test"),
        )

    # ── round lifecycle ─────────────────────────────────────────────────────
    def set_parameters(self, package: Dict[str, Any]) -> None:
        self.id = package["client_id"]
        self.current_iter = package["current_iter"]
        self._load_private(self.id)
        self.model.load_state_dict(package["regular_model_params"], strict=False)
        if package["personal_model_params"]:
            self.model.load_state_dict(package["personal_model_params"], strict=False)
        if package["optimizer_state"]:
            self.optimizer.load_state_dict(package["optimizer_state"])
        else:
            self.optimizer.load_state_dict(self.init_optimizer_state)
        if package["scheduler_state"]:
            self.scheduler.load_state_dict(package["scheduler_state"])
        else:
            self.scheduler.load_state_dict(self.init_scheduler_state)

    def fit(self) -> None:
        SharedMethods._set_worker_seed(self._loader_seed("train"))
        loader = self.load_train_data()
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
        self.set_parameters(package)
        start = time.time()
        self.fit()
        return self.package(train_time=time.time() - start)

    def package(self, train_time: float) -> Dict[str, Any]:
        state = self.model.state_dict()
        regular = {
            k: state[k].detach().cpu().clone() for k in self.regular_params_name
        }
        personal = {
            k: state[k].detach().cpu().clone() for k in self.personal_params_name
        }
        return {
            "client_id": self.id,
            "regular_model_params": regular,
            "personal_model_params": personal,
            "optimizer_state": self._optimizer_state_to_cpu(self.optimizer),
            "scheduler_state": copy.deepcopy(self.scheduler.state_dict()),
            "score": self.train_samples,
            "train_time": train_time,
        }

    def evaluate_global(
        self,
        client_id: int,
        global_params: "OrderedDict[str, torch.Tensor]",
        dataset_type: str,
        current_iter: int,
    ) -> float:
        self.id = client_id
        self.current_iter = current_iter
        self._load_private(client_id)
        self.model.load_state_dict(global_params, strict=False)
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

    def evaluate_personalized(
        self,
        client_id: int,
        global_params: "OrderedDict[str, torch.Tensor]",
        personal_params: Dict[str, torch.Tensor],
        dataset_type: str,
        current_iter: int,
    ) -> float:
        """Evaluate the per-client model (global overlaid with personal params)."""
        self.id = client_id
        self.current_iter = current_iter
        self._load_private(client_id)
        self.model.load_state_dict(global_params, strict=False)
        if personal_params:
            self.model.load_state_dict(personal_params, strict=False)
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


class Trainer:
    """Drives per-client work serially or across a Ray actor pool."""

    def __init__(self, server: "StatelessServer", client_cls, configs, times) -> None:
        self.server = server
        self.client_cls = client_cls
        self.parallel = server.parallel
        if not self.parallel:
            self.worker = client_cls(configs=configs, times=times, device=configs.device)
        else:
            self.num_workers = int(server.num_workers)
            device = "cuda" if server.num_gpus > 0 else "cpu"
            remote_cls = ray.remote(
                num_gpus=server.num_gpus / self.num_workers
            )(client_cls)
            self.workers = [
                remote_cls.remote(configs=configs, times=times, device=device)
                for _ in range(self.num_workers)
            ]

    def train(self, selected: List[int]) -> "OrderedDict[int, dict]":
        packages: "OrderedDict[int, dict]" = OrderedDict()
        if not self.parallel:
            for cid in selected:
                out = self.worker.train(self.server.package(cid))
                self._write_back(cid, out)
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
                fut = self.workers[wid].train.remote(self.server.package(cid))
                job_map[fut] = (cid, wid)
                futures.append(fut)
                i += 1
            if futures:
                done, futures = ray.wait(futures)
                for fut in done:
                    cid, wid = job_map.pop(fut)
                    out = ray.get(fut)
                    self._write_back(cid, out)
                    results[cid] = out
                    idle.append(wid)
        for cid in selected:
            packages[cid] = results[cid]
        return packages

    def evaluate(
        self, ids: List[int], global_params, dataset_type: str, current_iter: int
    ) -> List[float]:
        if not self.parallel:
            return [
                self.worker.evaluate_global(
                    cid, global_params, dataset_type, current_iter
                )
                for cid in ids
            ]
        gp = ray.put(global_params)
        futures = [
            self.workers[k % self.num_workers].evaluate_global.remote(
                cid, gp, dataset_type, current_iter
            )
            for k, cid in enumerate(ids)
        ]
        return list(ray.get(futures))

    def evaluate_personalized(
        self, ids: List[int], global_params, personal_map, dataset_type, current_iter
    ) -> List[float]:
        if not self.parallel:
            return [
                self.worker.evaluate_personalized(
                    cid, global_params, personal_map[cid], dataset_type, current_iter
                )
                for cid in ids
            ]
        gp = ray.put(global_params)
        futures = [
            self.workers[k % self.num_workers].evaluate_personalized.remote(
                cid, gp, personal_map[cid], dataset_type, current_iter
            )
            for k, cid in enumerate(ids)
        ]
        return list(ray.get(futures))

    def _write_back(self, cid: int, out: Dict[str, Any]) -> None:
        self.server.client_optimizer_states[cid] = out["optimizer_state"]
        self.server.client_scheduler_states[cid] = out["scheduler_state"]
        self.server.clients_personal_model_params[cid].update(
            out["personal_model_params"]
        )


class StatelessServer(SharedMethods):
    """Server that owns all per-client state and aggregates a global model."""

    def __init__(self, configs: Namespace, times: int) -> None:
        self.set_configs(configs=configs, times=times)
        self.mkdir()
        self.current_iter = 0
        self.new_client_gen_test_loss: Optional[float] = None
        self.new_client_pers_test_loss: Optional[float] = None
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
            "global_avg_train_loss": [],
            "personal_avg_train_loss": [],
            "global_avg_test_loss": [],
            "personal_avg_test_loss": [],
            "send_mb": [],
        }
        self.make_logger(name=self.name, path=self.log_path)

        with open(self.path_info, "r", encoding="utf-8") as f:
            info0 = json.load(f)[0]
        self.configs.__dict__["input_channels"] = info0["input_channels"]
        self.input_channels = info0["input_channels"]
        self.configs.__dict__["output_channels"] = info0["output_channels"]
        self.output_channels = info0["output_channels"]

        model_cls = SharedMethods._get_objective_function("models", self.model)
        if _PARITY_RNG_SHIM:
            for _ in range(self.num_clients):
                model_cls(configs=self.configs)  # advance torch RNG to match legacy
        self.model = model_cls(configs=self.configs)
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
            new_ids = set(
                rng.choice(self.num_clients, num_new, replace=False).tolist()
            )
            for cid in new_ids:
                self.is_new[cid] = True
            self.logger.info(f"New clients ({num_new}): {sorted(new_ids)}")

        self.trainer = Trainer(self, self._client_cls(), self.configs, self.times)
        self._send_mb_per_round = self.num_clients * self.get_size(self.model)

    def _client_cls(self):
        module_name = self.__module__
        class_name = self.__class__.__name__ + "_Client"
        try:
            return getattr(
                __import__(module_name, fromlist=[class_name]), class_name
            )
        except (ImportError, AttributeError):
            return StatelessClient

    # ── round mechanics ─────────────────────────────────────────────────────
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

    def package(self, client_id: int) -> Dict[str, Any]:
        return {
            "client_id": client_id,
            "current_iter": self.current_iter,
            "regular_model_params": copy.deepcopy(self.public_model_params),
            "personal_model_params": self.clients_personal_model_params[client_id],
            "optimizer_state": self.client_optimizer_states[client_id],
            "scheduler_state": self.client_scheduler_states[client_id],
        }

    def _commit_global(self, new_params) -> None:
        """Set the global model params and load them into ``self.model``."""
        self.public_model_params = OrderedDict(new_params)
        self.model.load_state_dict(self.public_model_params, strict=False)

    def aggregate_client_updates(self, packages: "OrderedDict[int, dict]") -> None:
        scores = [p["score"] for p in packages.values()]
        total = float(sum(scores))
        weights = torch.tensor([s / total for s in scores], dtype=torch.float32)
        new_params = OrderedDict()
        for name in self.public_model_params:
            stacked = torch.stack(
                [p["regular_model_params"][name] for p in packages.values()], dim=-1
            )
            new_params[name] = torch.sum(stacked * weights.to(stacked.dtype), dim=-1)
        self._commit_global(new_params)

    def _pre_eval_hook(self, dataset_type: str) -> None:
        """Hook run before each round's local training. No-op for tFL-style
        strategies; pFL-style servers override it to evaluate personalization."""

    def evaluate_generalization(self, dataset_type: str) -> None:
        incumbent = [i for i in range(self.num_clients) if not self.is_new[i]]
        losses = self.trainer.evaluate(
            incumbent, self.public_model_params, dataset_type, self.current_iter
        )
        metric = f"global_avg_{dataset_type}_loss"
        self.metrics[metric].append(float(np.mean(losses)))
        self.logger.info(
            f"Generalization {dataset_type.capitalize()} Loss: "
            f"{self.metrics[metric][-1]:.4f}"
        )

    def early_stopping(self) -> bool:
        metric = self.metrics["global_avg_test_loss"]
        if not self.patience or len(metric) < self.patience:
            return False
        if min(metric) not in metric[-self.patience :]:
            self.logger.info("Early stopping activated.")
            return True
        return False

    def train(self) -> None:
        for i in range(self.iterations):
            round_start = time.time()
            self.current_iter = i
            self.logger.info("")
            self.logger.info(
                f"-------------Round number: {str(i).zfill(4)}-------------"
            )
            self.select_clients()
            self.metrics["send_mb"].append(self._send_mb_per_round)
            if i % self.eval_gap == 0:
                for dataset_type in ["train", "test"]:
                    if dataset_type == "train" and self.skip_eval_train:
                        continue
                    self._pre_eval_hook(dataset_type)
            packages = self.trainer.train(self.selected_clients)
            self.aggregate_client_updates(packages)
            if i % self.eval_gap == 0:
                for dataset_type in ["train", "test"]:
                    if dataset_type == "train" and self.skip_eval_train:
                        continue
                    if not self.exclude_server_model_processes:
                        self.evaluate_generalization(dataset_type)
            self.metrics["time_per_iter"].append(time.time() - round_start)
            self.fix_results(default=self.default_value)
            if self.early_stopping():
                break
        self.save_results()
        try:
            self.close_logger()
        except Exception:
            pass
        try:
            ray.shutdown()
        except Exception:
            pass


class StatelessPFLServer(StatelessServer):
    """Personalized-FL server.

    Adds a personalization evaluation before each round's local training: each
    client's model (the global model overlaid with its stored personal params)
    is evaluated on its own data. The personal-params round-trip itself is
    handled by the base ``StatelessServer``/``Trainer`` (any client that sets
    ``personal_params_name`` gets those params persisted per client).
    """

    def _pre_eval_hook(self, dataset_type: str) -> None:
        incumbent = [i for i in range(self.num_clients) if not self.is_new[i]]
        losses = self.trainer.evaluate_personalized(
            incumbent,
            self.public_model_params,
            self.clients_personal_model_params,
            dataset_type,
            self.current_iter,
        )
        metric = f"personal_avg_{dataset_type}_loss"
        self.metrics[metric].append(float(np.mean(losses)))
        self.logger.info(
            f"Personalization {dataset_type.capitalize()} Loss: "
            f"{self.metrics[metric][-1]:.4f}"
        )
