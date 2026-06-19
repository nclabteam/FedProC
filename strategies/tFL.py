import copy
import gc
import json
import logging
import time
from argparse import Namespace
from collections import deque
from concurrent.futures import ThreadPoolExecutor
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
                range(self.num_join_clients, len(incumbent_clients) + 1), 1, replace=False
            )[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        self.selected_clients = list(
            np.random.choice(incumbent_clients, self.current_num_join_clients, replace=False)
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
        current_iter = getattr(self, "current_iter", 0)
        for idx, client in enumerate(self.clients):
            client.current_iter = current_iter
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
        exclude_ratio = getattr(self, "exclude_ratio", 0.0)
        if exclude_ratio > 0.0:
            num_new = max(1, int(self.num_clients * exclude_ratio))
            rng = np.random.default_rng(getattr(self, "seed", 0))
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
        gen  = getattr(self, "new_client_gen_test_loss",  None)
        pers = getattr(self, "new_client_pers_test_loss", None)
        if gen is not None or pers is not None:
            import json, os
            path = os.path.join(self.save_path, "new_client_results.json")
            with open(path, "w") as f:
                json.dump({
                    "new_client_avg_gen_test_loss":  gen,
                    "new_client_avg_pers_test_loss": pers,
                }, f, indent=2)

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

    def evaluate_generalization_loss(self, dataset_type: str) -> None:
        incumbent_clients = [c for c in self.clients if not c.is_new]
        if self.parallel:
            futures = []
            for client in incumbent_clients:
                device = "cuda" if self.num_gpus > 0 else "cpu"
                num_gpus = 1 / self.num_workers if self.num_gpus > 0 else 0
                future = ray_compute_client_loss.options(num_gpus=num_gpus).remote(
                    client=client,
                    mode="generalization",
                    dataset_type=dataset_type,
                    model=self.model,
                    criterion=client.loss,
                    device=device,
                )
                futures.append(future)
            losses = ray.get(futures)
        else:
            losses = [
                float(
                    np.mean(
                        self.calculate_loss(
                            model=self.model,
                            dataloader=getattr(client, f"load_{dataset_type}_data")(),
                            criterion=client.loss,
                            device=client.device,
                        )
                    )
                )
                for client in incumbent_clients
            ]
        metric_name = f"global_avg_{dataset_type}_loss"
        self.metrics[metric_name].append(float(np.mean(losses)))
        self.logger.info(
            f"Generalization {dataset_type.capitalize()} Loss: "
            f"{self.metrics[metric_name][-1]:.4f}"
        )

    def evaluate_personalization_loss(self, dataset_type: str) -> None:
        incumbent_clients = [c for c in self.clients if not c.is_new]
        if self.parallel:
            futures = []
            for client in incumbent_clients:
                device = "cuda" if self.num_gpus > 0 else "cpu"
                num_gpus = 1 / self.num_workers if self.num_gpus > 0 else 0
                future = ray_compute_client_loss.options(num_gpus=num_gpus).remote(
                    client=client,
                    mode="personalization",
                    dataset_type=dataset_type,
                    device=device,
                )
                futures.append(future)
            losses = ray.get(futures)
        else:
            losses = [
                float(getattr(client, f"get_{dataset_type}_loss")())
                for client in incumbent_clients
            ]
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
            float(np.mean(self.calculate_loss(
                model=self.model,
                dataloader=client.load_test_data(),
                criterion=client.loss,
                device=client.device,
            )))
            for client in self.new_clients
        ]
        self.new_client_gen_test_loss = float(np.mean(losses))
        self.logger.info(f"New Client Gen  Test Loss: {self.new_client_gen_test_loss:.4f}")

    def evaluate_new_clients_pers(self) -> None:
        if not self.new_clients:
            return
        losses = [float(client.get_test_loss()) for client in self.new_clients]
        self.new_client_pers_test_loss = float(np.mean(losses))
        self.logger.info(f"New Client Pers Test Loss: {self.new_client_pers_test_loss:.4f}")

    def _run_clients(self, method: str, clients: List) -> None:
        """Dispatch ``client.<method>()`` across *clients* in parallel or serial.

        Use this in strategy overrides that call a client method other than
        ``train()`` (e.g. ``compute_statistics``, ``adaptive_local_aggregation``).
        Calling ``_run_clients`` instead of writing a bare ``for`` loop ensures
        the correct execution mode is always used without duplicating the
        parallel-dispatch boilerplate in every strategy.

        ``train()`` is excluded from this dispatcher: it returns a package dict
        that ``train_clients`` must apply back to the original client, so it
        keeps its own Ray-based parallel path.

        Parameters
        ----------
        method : str
            Name of the client method to invoke.  The method must take no
            positional arguments.  It may mutate client state in place; the
            return value is discarded.
        clients : list
            Client objects on which to dispatch ``method``.

        Notes
        -----
        Parallel mode uses :class:`concurrent.futures.ThreadPoolExecutor`.
        CPU-bound methods (matrix math via NumPy/PyTorch) release the GIL, so
        threads achieve genuine parallelism without Ray serialisation overhead.
        Because threads share the process address space, in-place state
        mutations (e.g. ``self._Sigma_xx``) are immediately visible on the
        original client objects — no round-trip serialisation required.

        Examples
        --------
        Replace a serial loop in a strategy override::

            # Before (serial, loses Ray):
            for client in clients:
                client.compute_statistics()

            # After (honours self.parallel):
            self._run_clients("compute_statistics", clients)
        """
        if not self.parallel or len(clients) <= 1:
            for client in clients:
                getattr(client, method)()
            return

        n_workers = min(len(clients), self.num_workers)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(getattr(client, method)) for client in clients]
            for future in futures:
                future.result()  # propagate exceptions; discard return value

    def train_clients(self, new_only: bool = False) -> None:
        clients = self.new_clients if new_only else self.selected_clients
        if new_only and isinstance(self.model, torch.nn.Module):
            for client in clients:
                client.update_model_params(old=client.model, new=self.model)
        current_iter = getattr(self, "current_iter", 0)
        if self.parallel:
            i = 0
            futures = []
            idle_workers = deque(range(self.num_workers))
            job_map = {}
            client_packages = {}
            while i < len(clients) or len(futures) > 0:
                while i < len(clients) and len(idle_workers) > 0:
                    worker_id = idle_workers.popleft()
                    client = clients[i]
                    client.current_iter = current_iter
                    future = ray.remote(num_gpus=self.num_gpus / self.num_workers)(
                        lambda cl: cl.train()
                    ).remote(client)
                    job_map[future] = (client, worker_id)
                    futures.append(future)
                    i += 1
                if len(futures) > 0:
                    all_finished, futures = ray.wait(futures)
                    for finished in all_finished:
                        client, worker_id = job_map[finished]
                        client_package = ray.get(finished)
                        idle_workers.append(worker_id)
                        client_packages[client] = client_package
                        if client_package is not None:
                            client.update_model_params(
                                old=client.model, new=client_package["model"]
                            )
                            client.update_optimizer_params(
                                old=client.optimizer,
                                new=client_package["optimizer_state"],
                            )
                            client.metrics["train_time"].append(
                                client_package["train_time"]
                            )
                            client.train_samples = client_package["train_samples"]
        else:
            for client in clients:
                client.current_iter = current_iter
                client.train()

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

    def _loader_seed(self, dataset_type: str) -> Optional[int]:
        base_seed = getattr(self, "seed", None)
        if base_seed is None:
            return None
        dataset_offset = {"train": 0, "test": 1, "valid": 2}.get(dataset_type, 3)
        return self._derive_seed(
            int(base_seed) + int(getattr(self, "times", 0)),
            getattr(self, "id", 0),
            getattr(self, "current_iter", 0),
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
        adapt_T = getattr(self, "adapt_T", None)
        if self.is_new and adapt_T is not None:
            trainloader = self.load_data_head(
                file=self.train_file,
                T=adapt_T,
                shuffle=shuffle,
                scaler=self.scaler,
                batch_size=self.batch_size,
            )
        else:
            if sample_ratio is None:
                sample_ratio = getattr(self, "sample_ratio", 1.0)
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

    def train(self) -> Optional[Dict[str, Any]]:
        seed = self._loader_seed("train") if hasattr(self, "_loader_seed") else None
        SharedMethods._set_worker_seed(seed)
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
        train_time = time.time() - start_time
        if self.parallel:
            model = self.model
            if self.efficiency == "high":
                model = self._clone_model_to_cpu(self.model)
            return {
                "id": self.id,
                "model": model,
                "optimizer_state": self._optimizer_state_to_cpu(self.optimizer),
                "train_time": train_time,
                "train_samples": self.train_samples,
            }
        self.metrics["train_time"].append(train_time)
        return None

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
            self.logger.warning("adapt called with no global model — starting from random init")
        adapt_epochs = getattr(self, "adapt_epochs", 1)
        train_loader = self.load_train_data()
        self.model.to(self.device)
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        for _ in range(adapt_epochs):
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
