import copy
import gc
import json
import logging
import time
from argparse import Namespace
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import ray
import torch
from torch.utils.data import DataLoader

from .base import SharedMethods, ray_compute_client_loss


class tFL(SharedMethods):
    """
    Traditional (centralized) Federated Learning.

    Server orchestrates client selection, model aggregation, communication,
    and experiment tracking. No client model saving or personalization —
    those belong in pFL.
    """

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__()
        self.set_configs(configs=configs, times=times)
        self.mkdir()

        self.current_iter: int = 0
        self.new_client_gen_test_loss: Optional[float] = None
        self.new_client_pers_test_loss: Optional[float] = None
        self.num_join_clients = max(1, int(self.num_clients * self.join_ratio))
        self.current_num_join_clients = self.num_join_clients

        device_ids = [device_id for device_id in self.device_id.split(",") if device_id]
        self.num_gpus = len(device_ids) if self.device == "cuda" else 0
        configs.parallel = True if self.num_gpus > 0 and self.num_workers > 0 else False
        self.parallel = configs.parallel
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
        self.initialize_clients()
        self.configs.__dict__["input_channels"] = self.clients[0].input_channels
        self.input_channels = self.clients[0].input_channels
        self.configs.__dict__["output_channels"] = self.clients[0].output_channels
        self.output_channels = self.clients[0].output_channels
        self.initialize_model()
        self.get_model_info()

    def select_clients(self) -> None:
        incumbent_clients = [c for c in self.clients if not c.is_new]
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(
                range(self.num_join_clients, len(incumbent_clients) + 1),
                1,
                replace=False,
            )[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        self.selected_clients = list(
            np.random.choice(
                incumbent_clients, self.current_num_join_clients, replace=False
            )
        )

    def variables_to_be_sent(self) -> Dict[str, Any]:
        return {"model": self.model}

    def send_to_clients(self) -> None:
        total_bytes_sent = 0.0
        to_be_sent = self.variables_to_be_sent()
        shared_sizes = {
            key: self.get_size(value)
            for key, value in to_be_sent.items()
            if not (isinstance(value, list) and len(value) == len(self.clients))
        }
        for idx, client in enumerate(self.clients):
            client.current_iter = self.current_iter
            data_to_send = {}
            for key, value in to_be_sent.items():
                if isinstance(value, list) and len(value) == len(self.clients):
                    value = value[idx]
                    total_bytes_sent += self.get_size(value)
                else:
                    total_bytes_sent += shared_sizes[key]
                data_to_send[key] = value
            client.receive_from_server(data_to_send)
        self.metrics["send_mb"].append(total_bytes_sent)

    def receive_from_clients(self) -> None:
        self.client_data = []
        for client in self.selected_clients:
            try:
                self.client_data.append(client.send_to_server())
            except Exception as e:
                self.logger.error(
                    f"Failed to receive data from client {client.id}: {e}"
                )

    def initialize_clients(self) -> None:
        module_name = self.__module__
        class_name = self.__class__.__name__ + "_Client"
        try:
            client_class = getattr(
                __import__(module_name, fromlist=[class_name]), class_name
            )
        except (ImportError, AttributeError):
            client_class = tFL_Client
        self.clients = [
            client_class(configs=self.configs, id=cid, times=self.times)
            for cid in range(self.num_clients)
        ]
        if self.exclude_ratio > 0.0:
            num_new = max(1, int(self.num_clients * self.exclude_ratio))
            rng = np.random.default_rng(self.seed)
            new_ids = set(rng.choice(self.num_clients, num_new, replace=False).tolist())
            for client in self.clients:
                if client.id in new_ids:
                    client.is_new = True
            self.logger.info(f"New clients ({num_new}): {sorted(new_ids)}")

    @property
    def new_clients(self):
        return [c for c in self.clients if c.is_new]

    def save_results(self) -> None:
        super().save_results()
        for client in self.clients:
            if not client.is_new:
                client.save_results()
        if (
            self.new_client_gen_test_loss is not None
            or self.new_client_pers_test_loss is not None
        ):
            import json
            import os

            path = os.path.join(self.save_path, "new_client_results.json")
            with open(path, "w") as f:
                json.dump(
                    {
                        "new_client_avg_gen_test_loss": self.new_client_gen_test_loss,
                        "new_client_avg_pers_test_loss": self.new_client_pers_test_loss,
                    },
                    f,
                    indent=2,
                )

    def save_models(self, save_type: str) -> None:
        if save_type not in ["last", "best"]:
            raise ValueError("save_type must be 'last' or 'best'")

        should_save = True
        if save_type == "best":
            metric_key = "global_avg_test_loss"
            if metric_key not in self.metrics or not self.metrics[metric_key]:
                should_save = False
            else:
                metric_values = self.metrics[metric_key]
                if metric_values[-1] != min(metric_values):
                    should_save = False

        if not should_save:
            return

        if not self.exclude_server_model_processes:
            self.save_model(
                model=self.model,
                path=self.model_path,
                name=self.name,
                postfix=save_type,
                configs=self.configs,
                metadata={"save_type": save_type, "owner": "server"},
                verbose=self.logger,
            )

    def early_stopping(self) -> bool:
        metric = self.metrics["global_avg_test_loss"]
        if not self.patience or len(metric) < self.patience:
            return False
        best_so_far = min(metric)
        if best_so_far not in metric[-self.patience :]:
            self.logger.info("Early stopping activated.")
            return True
        return False

    def _gather_losses(self, mode: str, dataset_type: str) -> List[float]:
        """Compute one scalar loss per incumbent client, serial or parallel.

        The map-reduce counterpart of :meth:`_dispatch`: it *gathers* a value
        from each client rather than applying state back.  Parallel mode reuses
        the same bounded Ray worker pool as :meth:`_dispatch` (one in-flight job
        per worker) instead of fanning out an unbounded number of futures.

        Parameters
        ----------
        mode : {"generalization", "personalization"}
            ``"generalization"`` evaluates the shared global model on each
            client's data; ``"personalization"`` evaluates each client's own
            model via its ``get_<dataset_type>_loss``.
        dataset_type : str
            Split to evaluate, e.g. ``"train"`` or ``"test"``.

        Returns
        -------
        list of float
            One mean loss per incumbent (non-new) client, in client order.
        """
        incumbent_clients = [c for c in self.clients if not c.is_new]
        if not self.parallel:
            if mode == "generalization":
                return [
                    float(
                        np.mean(
                            self.calculate_loss(
                                model=self.model,
                                dataloader=getattr(
                                    client, f"load_{dataset_type}_data"
                                )(),
                                criterion=client.loss,
                                device=client.device,
                            )
                        )
                    )
                    for client in incumbent_clients
                ]
            return [
                float(getattr(client, f"get_{dataset_type}_loss")())
                for client in incumbent_clients
            ]

        device = "cuda" if self.num_gpus > 0 else "cpu"
        num_gpus = 1 / self.num_workers if self.num_gpus > 0 else 0
        losses: List[Optional[float]] = [None] * len(incumbent_clients)
        i = 0
        futures = []
        idle_workers = deque(range(self.num_workers))
        job_map: Dict[Any, tuple] = {}
        while i < len(incumbent_clients) or futures:
            while i < len(incumbent_clients) and idle_workers:
                worker_id = idle_workers.popleft()
                client = incumbent_clients[i]
                call_kwargs = dict(
                    client=client,
                    mode=mode,
                    dataset_type=dataset_type,
                    device=device,
                )
                if mode == "generalization":
                    call_kwargs["model"] = self.model
                    call_kwargs["criterion"] = client.loss
                future = ray_compute_client_loss.options(num_gpus=num_gpus).remote(
                    **call_kwargs
                )
                job_map[future] = (i, worker_id)
                futures.append(future)
                i += 1
            if futures:
                done, futures = ray.wait(futures)
                for f in done:
                    idx, worker_id = job_map.pop(f)
                    losses[idx] = float(ray.get(f))
                    idle_workers.append(worker_id)
        return [loss for loss in losses]

    def evaluate_generalization_loss(self, dataset_type: str) -> None:
        losses = self._gather_losses("generalization", dataset_type)
        metric_name = f"global_avg_{dataset_type}_loss"
        self.metrics[metric_name].append(float(np.mean(losses)))
        self.logger.info(
            f"Generalization {dataset_type.capitalize()} Loss: "
            f"{self.metrics[metric_name][-1]:.4f}"
        )

    def evaluate_personalization_loss(self, dataset_type: str) -> None:
        losses = self._gather_losses("personalization", dataset_type)
        metric_name = f"personal_avg_{dataset_type}_loss"
        self.metrics[metric_name].append(float(np.mean(losses)))
        self.logger.info(
            f"Personalization {dataset_type.capitalize()} Loss: "
            f"{self.metrics[metric_name][-1]:.4f}"
        )

    def adapt_new_clients(self) -> None:
        pass

    def evaluate_new_clients_gen(self) -> None:
        if not self.new_clients:
            return
        if not isinstance(self.model, torch.nn.Module):
            return
        losses = [
            float(
                np.mean(
                    self.calculate_loss(
                        model=self.model,
                        dataloader=client.load_test_data(),
                        criterion=client.loss,
                        device=client.device,
                    )
                )
            )
            for client in self.new_clients
        ]
        self.new_client_gen_test_loss = float(np.mean(losses))
        self.logger.info(
            f"New Client Gen  Test Loss: {self.new_client_gen_test_loss:.4f}"
        )

    def evaluate_new_clients_pers(self) -> None:
        if not self.new_clients:
            return
        losses = [float(client.get_test_loss()) for client in self.new_clients]
        self.new_client_pers_test_loss = float(np.mean(losses))
        self.logger.info(
            f"New Client Pers Test Loss: {self.new_client_pers_test_loss:.4f}"
        )

    _KNOWN_PACKAGE_KEYS: frozenset = frozenset(
        {
            "model",
            "optimizer_state",
            "train_time",
            "train_samples",
            "id",
        }
    )

    def _apply_client_result(self, client, package: Optional[Dict[str, Any]]) -> None:
        """Apply the dict returned by ``client.train()`` back to *client*.

        Called in both serial and parallel mode so there is a single write-back
        path.  In serial mode the client object is the same in-memory instance
        as during training, so identity checks make the copy operations cheap
        no-ops.  In parallel mode Ray deserialises a fresh copy, the checks
        fail, and the real copies happen.

        Strategies whose ``train()`` returns keys other than the gradient
        default **must** override this method — the base implementation raises
        :exc:`NotImplementedError` on unrecognised keys to surface the contract
        violation at runtime rather than silently dropping state.

        Parameters
        ----------
        client : tFL_Client
            The original (non-Ray) client object to update.
        package : dict or None
            Return value of ``client.train()``.  ``None`` is a no-op.

        Raises
        ------
        NotImplementedError
            If *package* contains keys not in
            ``{"model", "optimizer_state", "train_time", "train_samples", "id"}``.

        Examples
        --------
        Override in a one-shot statistics strategy::

            def _apply_client_result(self, client, package):
                if package is None:
                    return
                client._sigma_xx = package["sigma_xx"]
                client._sigma_xy = package["sigma_xy"]
                client.train_samples = package["train_samples"]
        """
        if package is None:
            return
        unknown = set(package) - self._KNOWN_PACKAGE_KEYS  # type: ignore[operator]
        if unknown:
            raise NotImplementedError(
                f"{type(self).__name__} received unknown package keys "
                f"{sorted(unknown)} from {type(client).__name__}.train() — "
                f"add them to {type(self).__name__}._KNOWN_PACKAGE_KEYS and "
                f"override _apply_client_result() to handle them."
            )
        model = package.get("model")
        if model is not None and model is not client.model:
            client.update_model_params(old=client.model, new=model)
        opt = package.get("optimizer_state")
        if opt is not None and opt is not client.optimizer:
            client.update_optimizer_params(old=client.optimizer, new=opt)
        if "train_time" in package:
            client.metrics["train_time"].append(package["train_time"])
        if "train_samples" in package:
            client.train_samples = package["train_samples"]

    def _dispatch(self, method: str, clients: List) -> None:
        """Dispatch ``client.<method>()`` across *clients*, serial or parallel.

        This is the single, unified execution path for all per-client method
        calls.  Strategies never write their own serial loops or parallel
        boilerplate — they call ``_dispatch`` and the framework decides the
        execution mode based on ``self.parallel``.

        Parallel mode uses the same Ray worker-pool pattern for every method,
        regardless of which strategy or client class is involved.  The return
        value of ``method`` is forwarded to
        :meth:`_apply_client_result` so that state produced on the remote copy
        is written back to the original client.

        Parameters
        ----------
        method : str
            Name of the no-argument client method to invoke.
        clients : list
            Client objects on which to dispatch ``method``.

        Notes
        -----
        ``train_clients`` is the primary caller; strategies that need to
        dispatch a *different* method (e.g. ``adaptive_local_aggregation``)
        call ``_dispatch`` directly with the appropriate method name instead of
        writing a loop.

        Examples
        --------
        Dispatch a strategy-specific client method in parallel::

            def some_server_step(self):
                self._dispatch("adaptive_local_aggregation", self.selected_clients)
        """
        if not self.parallel:
            for client in clients:
                self._apply_client_result(client, getattr(client, method)())
            return

        i = 0
        futures = []
        idle_workers = deque(range(self.num_workers))
        job_map: Dict[Any, tuple] = {}

        while i < len(clients) or futures:
            while i < len(clients) and idle_workers:
                worker_id = idle_workers.popleft()
                client = clients[i]
                future = ray.remote(num_gpus=self.num_gpus / self.num_workers)(
                    lambda cl, m: cl._offload_package(getattr(cl, m)())
                ).remote(client, method)
                job_map[future] = (client, worker_id)
                futures.append(future)
                i += 1
            if futures:
                done, futures = ray.wait(futures)
                for f in done:
                    client, worker_id = job_map.pop(f)
                    self._apply_client_result(client, ray.get(f))
                    idle_workers.append(worker_id)

    def train_clients(self, new_only: bool = False) -> None:
        """Train all selected clients, in serial or parallel.

        Parameters
        ----------
        new_only : bool, optional
            When ``True`` operate on :attr:`new_clients` instead of
            :attr:`selected_clients` and seed each new client with the current
            global model before training.
        """
        clients = self.new_clients if new_only else self.selected_clients
        if new_only and isinstance(self.model, torch.nn.Module):
            for client in clients:
                client.update_model_params(old=client.model, new=self.model)
        for client in clients:
            client.current_iter = self.current_iter
        self._dispatch("train", clients)

    def fix_results(self) -> None:
        super().fix_results(default=self.default_value)
        for client in self.clients:
            client.fix_results(default=self.default_value)

    def calculate_aggregation_weights(self) -> None:
        ts = [client["score"] for client in self.client_data]
        self.weights = torch.tensor(ts) / sum(ts)

    def aggregate_models(self) -> None:
        if self.return_diff:
            for client, weight in zip(self.client_data, self.weights):
                for global_param, local_param in zip(
                    self.model.parameters(), client["model"].parameters()
                ):
                    global_param.data.sub_(
                        local_param.data.to(global_param.device), alpha=weight
                    )
        else:
            self.model = self.reset_model(self.model)
            for client, weight in zip(self.client_data, self.weights):
                for global_param, local_param in zip(
                    self.model.parameters(), client["model"].parameters()
                ):
                    global_param.data.add_(
                        local_param.data.to(global_param.device), alpha=weight
                    )

    def get_model_info(self) -> None:
        if not self.exclude_server_model_processes:
            dl = self.clients[0].load_train_data()
            self.summarize_model(dataloader=dl)
            del dl
            gc.collect()

    def post_process(self) -> None:
        self.logger.info("")
        self.logger.info("-" * 50)
        self.evaluate_new_clients_gen()
        self.train_clients(new_only=True)
        self.adapt_new_clients()
        self.evaluate_new_clients_pers()
        self.save_models(save_type="last")
        self.save_results()
        for c in self.clients:
            try:
                c.close_logger()
            except Exception:
                pass
        try:
            self.close_logger()
        except Exception:
            pass
        try:
            ray.shutdown()
        except Exception:
            pass

    def pre_train_clients(self) -> None:
        pass

    def _pre_eval_hook(self, dataset_type: str) -> None:
        pass

    def train(self) -> None:
        for i in range(self.iterations):
            round_start_time = time.time()
            self.current_iter = i
            self.logger.info("")
            self.logger.info(
                f"-------------Round number: {str(self.current_iter).zfill(4)}-------------"
            )
            self.select_clients()
            self.send_to_clients()
            if self.current_iter % self.eval_gap == 0:
                for dataset_type in ["train", "test"]:
                    if dataset_type == "train" and self.skip_eval_train:
                        continue
                    self._pre_eval_hook(dataset_type)
            self.pre_train_clients()
            self.train_clients()
            self.receive_from_clients()
            self.calculate_aggregation_weights()
            self.aggregate_models()
            if self.current_iter % self.eval_gap == 0:
                for dataset_type in ["train", "test"]:
                    if dataset_type == "train" and self.skip_eval_train:
                        continue
                    if not self.exclude_server_model_processes:
                        self.evaluate_generalization_loss(dataset_type)
            self.save_models(save_type="best")
            round_duration = time.time() - round_start_time
            self.metrics["time_per_iter"].append(round_duration)
            self.logger.info(f"Time cost: {round_duration:.4f}s")
            self.fix_results()
            if self.early_stopping():
                break
        self.post_process()


class tFL_Client(SharedMethods):
    def __init__(self, configs: Namespace, id: int, times: int) -> None:
        super().__init__()
        self.set_configs(configs=configs, id=id, times=times)
        self.mkdir()
        self.initialize_private_info()
        self.initialize_model()
        self.initialize_loss()
        self.initialize_optimizer()
        self.initialize_scheduler()
        self.initialize_scaler()
        self.current_iter: int = 0
        self.name = f"CLIENT_{str(self.id).zfill(3)}"
        self.make_logger(name=self.name, path=self.log_path)
        self.is_new = False
        self.metrics = {
            "train_time": [],
            "train_loss": [],
            "test_loss": [],
            "send_mb": [],
        }

    def initialize_private_info(self) -> None:
        with open(self.path_info, "r", encoding="utf-8") as f:
            self.private_data = json.load(f)[self.id]
        if self.private_data["client"] != self.id:
            raise ValueError("Client ID mismatch")
        self.train_file = self.private_data["paths"]["train"]
        self.test_file = self.private_data["paths"]["test"]
        self.stats = self.private_data["stats"]["train"]
        self.configs.__dict__["input_channels"] = self.private_data["input_channels"]
        self.input_channels = self.private_data["input_channels"]
        self.configs.__dict__["output_channels"] = self.private_data["output_channels"]
        self.output_channels = self.private_data["output_channels"]

    def initialize_scaler(self) -> None:
        self.scaler = getattr(__import__("scalers"), self.scaler)(self.stats)

    def variables_to_be_sent(self) -> Dict[str, Any]:
        if self.return_diff:
            diff_dict = {
                key: param_old - param_new.detach().to("cpu")
                for (key, param_old), param_new in zip(
                    self.snapshot.named_parameters(), self.model.parameters()
                )
            }
            diff_model = copy.deepcopy(self.model)
            diff_model.load_state_dict(diff_dict)
            model = diff_model
            del self.snapshot
        else:
            model = self.model
        if self.efficiency == "high":
            model = self._clone_model_to_cpu(model)
        return {"model": model, "score": self.train_samples}

    def _clone_model_to_cpu(self, model: torch.nn.Module) -> torch.nn.Module:
        clone = copy.deepcopy(model)
        clone.train(model.training)
        return clone.to("cpu")

    def _offload_package(
        self, package: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Move the standard model/optimizer entries of a train package to CPU.

        Called only on the Ray worker path (see :meth:`tFL._dispatch`) so that
        GPU tensors are converted before crossing the process boundary back to
        the driver.  Serial dispatch never calls this, so live GPU references
        are preserved untouched.  Strategy-specific keys are left as-is —
        strategies that add custom tensors handle their own device placement
        (or override this method).

        Parameters
        ----------
        package : dict or None
            The dict returned by a dispatched client method.

        Returns
        -------
        dict or None
            The same package with ``model`` moved to CPU and
            ``optimizer_state`` converted to a CPU state dict, when present.
        """
        if package is None:
            return None
        model = package.get("model")
        if isinstance(model, torch.nn.Module):
            package["model"] = model.to("cpu")
        optimizer = package.get("optimizer_state")
        if isinstance(optimizer, torch.optim.Optimizer):
            package["optimizer_state"] = self._optimizer_state_to_cpu(optimizer)
        return package

    def _loader_seed(self, dataset_type: str) -> Optional[int]:
        if self.seed is None:
            return None
        dataset_offset = {"train": 0, "test": 1, "valid": 2}.get(dataset_type, 3)
        return self._derive_seed(
            int(self.seed) + int(self.times),
            self.id,
            self.current_iter,
            dataset_offset,
        )

    def send_to_server(self) -> Dict[str, Any]:
        to_be_sent = self.variables_to_be_sent()
        total_size = sum(self.get_size(value) for value in to_be_sent.values())
        self.metrics["send_mb"].append(total_size)
        return to_be_sent

    def load_train_data(
        self, sample_ratio: float = None, shuffle: bool = True
    ) -> DataLoader:
        if self.is_new and self.adapt_T is not None:
            trainloader = self.load_data_head(
                file=self.train_file,
                T=self.adapt_T,
                shuffle=shuffle,
                scaler=self.scaler,
                batch_size=self.batch_size,
            )
        else:
            if sample_ratio is None:
                sample_ratio = self.sample_ratio
            trainloader = self.load_data(
                file=self.train_file,
                sample_ratio=sample_ratio,
                shuffle=shuffle,
                scaler=self.scaler,
                batch_size=self.batch_size,
                seed=self._loader_seed("train"),
            )
        self.train_samples = len(trainloader.dataset)
        return trainloader

    def load_test_data(
        self, sample_ratio: float = None, shuffle: bool = False
    ) -> DataLoader:
        # The test set is the fixed generalization ruler and must ALWAYS be full.
        # --sample_ratio is a train-side starvation knob (e.g. the grokking probe);
        # subsampling test would make test losses incomparable across a sample_ratio
        # sweep. Force 1.0 regardless of the passed/instance value. Standard runs use
        # 1.0 anyway, so this only changes sub-1.0 train-starved runs.
        sample_ratio = 1.0
        testloader = self.load_data(
            file=self.test_file,
            sample_ratio=sample_ratio,
            shuffle=shuffle,
            scaler=self.scaler,
            batch_size=self.batch_size,
            seed=self._loader_seed("test"),
        )
        self.test_samples = len(testloader.dataset)
        return testloader

    def train(self) -> Dict[str, Any]:
        """Run local training for one federation round.

        Returns live references to the trained model and optimizer in all
        modes — it never checks ``efficiency`` or ``parallel``.
        :meth:`tFL._dispatch` applies the package via
        :meth:`tFL._apply_client_result` in both serial and parallel mode: in
        serial the references are identical to the client's own attributes so
        the apply is a true no-op; in parallel the Ray worker CPU-ifies the
        package (see :meth:`_offload_package`) and the driver copies the values
        back onto the original client.

        ``efficiency`` still governs device residency *during* training (the
        per-epoch offload for ``"low"`` and the post-training offload for
        ``"med"``); it no longer affects what ``train`` returns.

        Returns
        -------
        dict
            ``{"model", "optimizer_state", "train_time", "train_samples"}``

        Notes
        -----
        Strategies whose ``train()`` returns *additional* keys (e.g. closed-form
        statistics) must extend the server's :attr:`tFL._KNOWN_PACKAGE_KEYS` and
        override :meth:`tFL._apply_client_result` — the base implementation
        raises :exc:`NotImplementedError` on unknown keys.
        """
        SharedMethods._set_worker_seed(self._loader_seed("train"))
        train_loader = self.load_train_data()
        start_time = time.time()
        offload_after_epoch = self.efficiency == "low"
        for _ in range(self.epochs):
            self.train_one_epoch(
                model=self.model,
                dataloader=train_loader,
                optimizer=self.optimizer,
                criterion=self.loss,
                scheduler=self.scheduler,
                device=self.device,
                offload_after=offload_after_epoch,
            )
        if self.efficiency == "med":
            self.model.to("cpu")
        return {
            "model": self.model,
            "optimizer_state": self.optimizer,
            "train_time": time.time() - start_time,
            "train_samples": self.train_samples,
        }

    def get_train_loss(self) -> float:
        losses = self.calculate_loss(
            model=self.model,
            dataloader=self.load_train_data(),
            criterion=self.loss,
            device=self.device,
            offload_after=self.efficiency != "high",
        )
        losses = np.mean(losses)
        self.metrics["train_loss"].append(losses)
        return losses

    def get_test_loss(self):
        losses = self.calculate_loss(
            model=self.model,
            dataloader=self.load_test_data(),
            criterion=self.loss,
            device=self.device,
            offload_after=self.efficiency != "high",
        )
        losses = np.mean(losses)
        self.metrics["test_loss"].append(losses)
        return losses

    def adapt(self, global_model: Optional[torch.nn.Module]) -> None:
        if global_model is not None:
            self.update_model_params(old=self.model, new=global_model)
        else:
            self.logger.warning(
                "adapt called with no global model — starting from random init"
            )
        train_loader = self.load_train_data()
        self.model.to(self.device)
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        for _ in range(self.adapt_epochs):
            for batch_x, batch_y, x_mark, y_mark in train_loader:
                batch_x = batch_x.to(self.device, dtype=torch.float32)
                batch_y = batch_y.to(self.device, dtype=torch.float32)
                x_mark = x_mark.to(self.device, dtype=torch.float32)
                y_mark = y_mark.to(self.device, dtype=torch.float32)
                opt.zero_grad()
                out = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
                self.loss(out, batch_y).backward()
                opt.step()
        self.model.to("cpu")

    def receive_from_server(self, data):
        if "current_iter" in data:
            self.current_iter = data["current_iter"]
        if self.return_diff:
            self.snapshot = copy.deepcopy(data["model"]).to("cpu")
        self.update_model_params(old=self.model, new=data["model"])
