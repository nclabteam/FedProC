import copy
import gc
import json
import logging
import os
import sys
import time
from argparse import Namespace
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import polars as pl
import ray
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset


class SharedMethods:
    """
    Collection of small reusable utilities used by Server and Client classes.

    Methods are mostly stateless and operate on models, dataloaders and tensors.
    They are implemented as @staticmethod so they can be invoked without creating
    an instance when needed (e.g., utility scripts or tests).
    """

    default_value = 9_999_999.0

    @staticmethod
    def load_data(
        file: str,
        sample_ratio: float = 1.0,
        batch_size: int = 32,
        shuffle: bool = False,
        scaler: Any = None,
    ) -> DataLoader:
        """
        Loads data from a numpy .npz file and constructs a PyTorch DataLoader.

        Expects the file to contain arrays 'x' and 'y'. Applies `scaler.transform`
        to both x and y, converts to torch.float32 tensors, and returns a DataLoader.

        Args:
            file: Path to a .npz file containing arrays 'x' and 'y'.
            sample_ratio: Fraction of dataset to use (0.0 to 1.0). Defaults to 1.0.
            batch_size: The batch size for the DataLoader. Defaults to 32.
            shuffle: Whether to shuffle the dataset. Defaults to False.
            scaler: An object exposing a `.transform(array)` method for normalization.

        Returns:
            A PyTorch DataLoader containing the processed dataset.

        Raises:
            AssertionError: If sample_ratio is not within [0, 1].
        """
        assert 0 <= sample_ratio <= 1, "sample_ratio must be between 0 and 1"

        with np.load(file) as data:
            x = data["x"]
            y = data["y"]
        x = scaler.transform(x)
        y = scaler.transform(y)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(x, y)

        # Apply subsampling if necessary
        if sample_ratio < 1.0:
            subset_size = int(len(dataset) * sample_ratio)
            indices = torch.randperm(len(dataset))[:subset_size]
            dataset = Subset(dataset, indices)

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    @staticmethod
    def calculate_loss(
        model: torch.nn.Module,
        dataloader: DataLoader,
        criterion: Callable,
        device: Union[str, torch.device] = "cpu",
    ) -> List[float]:
        """
        Calculates loss for each batch in the dataloader without gradients.

        Args:
            model: The neural network model.
            dataloader: The experimental data loader.
            criterion: The loss function (e.g., nn.MSELoss).
            device: The device to perform computation on.

        Returns:
            A list of scalar loss values for each batch.
        """
        losses = []
        model.to(device)
        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                losses.append(loss.item())
        model.to("cpu")
        return losses

    @staticmethod
    def save_model(
        model: torch.nn.Module,
        path: str,
        name: str,
        postfix: str = "",
        extention: str = "pt",
        verbose: Optional[logging.Logger] = None,
    ) -> None:
        """
        Saves a PyTorch model to a specified directory.

        Args:
            model: The model instance to save.
            path: Target directory path.
            name: Base name of the model.
            postfix: Optional string to append to the filename.
            extention: File extension (default: "pt").
            verbose: Optional logger to record the save path.
        """
        save_path = os.path.join(
            path,
            f"{'_'.join([name.lower().strip(), postfix])}.{extention}",
        )
        torch.save(obj=model, f=save_path)
        if verbose is not None:
            verbose.info(f"Model saved to {save_path}")

    @staticmethod
    def reset_model(model: torch.nn.Module) -> torch.nn.Module:
        """
        Creates a deep copy of the model with all parameters zeroed out.

        Args:
            model: The template model.

        Returns:
            A new model instance with identical architecture but zeroed weights.
        """
        result = copy.deepcopy(model)
        for param in result.parameters():
            param.data.zero_()
        return result

    @staticmethod
    def get_size(obj: Any) -> float:
        """
        Computes the approximate memory size of various objects in Megabytes (MB).

        Args:
            obj: The object to measure (Tensor, Module, DataLoader, etc.).

        Returns:
            The size of the object in MB.
        """
        if isinstance(obj, torch.Tensor):
            return obj.element_size() * obj.nelement() / (1024**2)
        if isinstance(obj, torch.nn.Module):
            total_size = sum(
                param.element_size() * param.nelement() for param in obj.parameters()
            )
            total_size += sum(
                buffer.element_size() * buffer.nelement() for buffer in obj.buffers()
            )
            return total_size / (1024**2)
        if isinstance(obj, DataLoader):
            # Estimate size of current dataset
            total_size = sum(
                sum(
                    item.element_size() * item.nelement()
                    for item in data
                    if isinstance(item, torch.Tensor)
                )
                for data in obj.dataset
            )
            return total_size / (1024**2)
        if isinstance(obj, TensorDataset):
            total_size = sum(
                tensor.element_size() * tensor.nelement() for tensor in obj.tensors
            )
            return total_size / (1024**2)
        if isinstance(obj, dict):
            total_size = sum(SharedMethods.get_size(value) for value in obj.values())
            return total_size / (1024**2)
        if isinstance(obj, list):
            total_size = sum(SharedMethods.get_size(item) for item in obj)
            return total_size / (1024**2)
        return sys.getsizeof(obj) / (1024**2)

    @staticmethod
    def train_one_epoch(
        model: torch.nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
        scheduler: Optional[Any],
        device: Union[str, torch.device],
    ) -> None:
        """
        Trains the model for one full epoch over the dataloader.

        Args:
            model: The neural network model.
            dataloader: Training data loader.
            optimizer: PyTorch optimizer instance.
            criterion: Loss function.
            scheduler: Optional learning rate scheduler.
            device: Computing device (CPU/GPU).
        """
        model.to(device)
        model.train()
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        model.to("cpu")
        scheduler.step()

    @staticmethod
    def update_model_params(old: torch.nn.Module, new: torch.nn.Module) -> None:
        """
        Copies parameters from one model to another.

        Args:
            old: The target model to be updated.
            new: The source model containing new weights.
        """
        for old_param, new_param in zip(old.parameters(), new.parameters()):
            old_param.data.copy_(new_param.data)

    @staticmethod
    def update_optimizer_params(
        old: torch.optim.Optimizer, new: torch.optim.Optimizer
    ) -> None:
        """
        Synchronizes hyperparameters and parameters between two optimizers.

        Args:
            old: The target optimizer.
            new: The source optimizer.
        """
        for old_group, new_group in zip(old.param_groups, new.param_groups):
            # Update all hyperparameters dynamically
            for key in new_group.keys():
                if key != "params":  # Skip updating "params" directly
                    old_group[key] = new_group[key]

            # Update the model parameters inside param_groups
            for old_param, new_param in zip(old_group["params"], new_group["params"]):
                old_param.data.copy_(new_param.data)

    @staticmethod
    def _get_objective_function(func_type: str, func_name: str) -> Callable:
        """
        Dynamically imports and returns a specified function/class.

        Args:
            func_type: The module name (e.g., 'losses', 'models').
            func_name: The function or class name within that module.

        Returns:
            The imported function or class object.
        """
        module = __import__(func_type, fromlist=[func_name])
        func = getattr(module, func_name)
        return func

    def initialize_loss(self) -> None:
        """Initializes the loss function based on the strategy configuration."""
        obj = self._get_objective_function("losses", self.loss)
        self.loss = obj()

    def initialize_model(self) -> None:
        """Initializes the model architecture based on the strategy configuration."""
        obj = self._get_objective_function("models", self.model)
        self.model = obj(configs=self.configs)

    def initialize_optimizer(self) -> None:
        """Initializes the optimizer based on the model and configuration."""
        obj = self._get_objective_function("optimizers", self.optimizer)
        self.optimizer = obj(params=self.model.parameters(), configs=self.configs)

    def initialize_scheduler(self) -> None:
        """Initializes the learning rate scheduler if specified in configuration."""
        obj = self._get_objective_function("schedulers", self.scheduler)
        self.scheduler = obj(optimizer=self.optimizer, configs=self.configs)

    def summarize_model(self, dataloader: DataLoader) -> None:
        """
        Generates a visual and textual summary of the model architecture.

        Args:
            dataloader: A sample dataloader to determine input shapes.
        """
        getattr(__import__("utils"), "ModelSummarizer")(
            model=self.model,
            dataloader=dataloader,
            save_path=os.path.join(
                self.model_info_path, f"{self.name.lower().strip()}.svg"
            ),
            device=self.device,
        ).execute()

    def save_results(self) -> None:
        """Exports the accumulated metrics to a CSV file."""
        pl_df = pl.DataFrame(self.metrics)
        path = os.path.join(self.result_path, self.name.lower().strip() + ".csv")
        pl_df.write_csv(path)
        self.logger.info(f"Results saved to {path}")

    def fix_results(self, default: float = -1.0) -> None:
        """
        Pads all metric lists to the same length to ensure table consistency.

        Args:
            default: The filler value for missing metrics. Defaults to -1.0.
        """
        max_length = max(len(lst) for lst in self.metrics.values())
        for key in self.metrics.keys():
            if len(self.metrics[key]) < max_length:
                self.metrics[key].extend(
                    [default] * (max_length - len(self.metrics[key]))
                )

    def make_logger(self, name: str, path: str) -> None:
        """
        Initializes and configures a unique logger for the class instance.

        Args:
            name: Base name for the logger.
            path: Directory path where the log file will be saved.
        """
        log_path = os.path.join(path, f"{name.lower().strip()}.log")

        # Create a unique logger name using the instance id
        logger_name = f"{name}_{self.times}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers
        if self.logger.hasHandlers():
            for h in list(self.logger.handlers):
                try:
                    h.flush()
                    h.close()
                except Exception:
                    pass
            self.logger.handlers.clear()
        # make sure logs don't propagate to root handler which might duplicate FDs
        self.logger.propagate = False

        # Create file and stream handlers
        file_handler = logging.FileHandler(log_path)
        stream_handler = logging.StreamHandler()

        # Set logging format
        formatter = logging.Formatter(f"%(asctime)s ~ {name} ~ %(message)s")
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

        self.logger.info(f"Logger created at {log_path}")

    def close_logger(self) -> None:
        """Flushes and closes all handlers associated with the logger."""
        if hasattr(self, "logger"):
            for h in list(self.logger.handlers):
                try:
                    h.flush()
                    h.close()
                except Exception:
                    pass
            self.logger.handlers.clear()

    def set_configs(self, configs: Namespace, **kwargs: Any) -> None:
        """
        Sets class attributes based on the provided configuration namespace.

        Args:
            configs: The configuration namespace from argparse.
            **kwargs: Additional overrides for configuration attributes.
        """
        if isinstance(configs, Namespace):
            for key, value in vars(configs).items():
                setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.configs = configs

    def mkdir(self) -> None:
        """Creates the necessary directory structure for experiments."""
        self.save_path = os.path.join(self.save_path, str(self.times))
        self.model_path = os.path.join(self.save_path, "models")
        self.model_info_path = os.path.join(self.save_path, "models_info")
        self.log_path = os.path.join(self.save_path, "logs")
        self.result_path = os.path.join(self.save_path, "results")
        for dir_path in [
            self.save_path,
            self.model_path,
            self.log_path,
            self.model_info_path,
            self.result_path,
        ]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)


# --- Ray Remote Function: unified client-side loss worker ---
@ray.remote
def ray_compute_client_loss(
    client: Any,
    mode: str,
    dataset_type: str,
    model: Optional[torch.nn.Module] = None,
    criterion: Optional[Callable] = None,
    device: Union[str, torch.device] = "cpu",
) -> float:
    """
    Unified Ray remote worker to compute client-side losses.

    Supports two modes:
    1. 'generalization': Evaluates a provided model on the client's local dataset.
    2. 'personalization': Instructs the client to compute loss using its internal model.

    Args:
        client: The client object instance.
        mode: The evaluation mode ('generalization' or 'personalization').
        dataset_type: The dataset split to use ('train', 'test', or 'valid').
        model: Optional model instance for generalization evaluation.
        criterion: Optional loss function for generalization evaluation.
        device: The device to perform computation on.

    Returns:
        The computed scalar loss value.

    Raises:
        ValueError: If an unsupported mode is provided.
    """
    if mode == "generalization":
        # evaluate the provided model on the client's data
        if model is None or criterion is None:
            raise ValueError("model and criterion are required for generalization mode")
        model = model.to(device)
        dataloader = getattr(client, f"load_{dataset_type}_data")()
        losses = []
        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                losses.append(float(loss.item()))
        return float(np.mean(losses))

    elif mode == "personalization":
        # let the client compute its own loss on its local model
        client.device = device
        return float(getattr(client, f"get_{dataset_type}_loss")())

    else:
        raise ValueError(f"Unsupported mode for ray_compute_client_loss: {mode}")


class Server(SharedMethods):
    """
    Orchestrates the federated learning process on the server side.

    Handles client selection, model aggregation, communication, and overall
    experiment tracking. Inherits from SharedMethods for utility functions.
    """

    def __init__(self, configs: Namespace, times: int) -> None:
        """Initializes the Server instance with provided configurations.

        Args:
            configs: Global configuration namespace.
            times: Experiment iteration/repetition index.
        """
        super().__init__()
        self.set_configs(configs=configs, times=times)
        self.mkdir()

        self.num_join_clients = max(1, int(self.num_clients * self.join_ratio))
        self.current_num_join_clients = self.num_join_clients

        self.num_gpus = len(self.device_id.split(","))
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
        """
        Selects a subset of clients to participate in the current round.

        Uses either a fixed join ratio or a random join ratio based on configs.
        """
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(
                range(self.num_join_clients, self.num_clients + 1), 1, replace=False
            )[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        self.selected_clients = list(
            np.random.choice(self.clients, self.current_num_join_clients, replace=False)
        )

    def variables_to_be_sent(self) -> Dict[str, Any]:
        """
        Defines the variables to be communicated from server to clients.

        Returns:
            A dictionary containing objects to be sent (e.g., global model).
        """
        return {"model": self.model}

    def send_to_clients(self) -> None:
        """
        Broadcasts global parameters to all clients.

        Tracks the amount of data (in MB) sent during this operation.
        """
        total_bytes_sent = 0.0
        to_be_sent = self.variables_to_be_sent()
        for idx, client in enumerate(self.clients):
            data_to_send = {}
            for key, value in to_be_sent.items():
                if isinstance(value, list) and len(value) == len(self.clients):
                    value = value[idx]
                total_bytes_sent += self.get_size(value)
                data_to_send[key] = value
            client.receive_from_server(data_to_send)
        self.metrics["send_mb"].append(total_bytes_sent)

    def receive_from_clients(self) -> None:
        """
        Collects updated parameters or statistics from the selected clients.

        Stores the collected data in `self.client_data`.
        """
        self.client_data = []
        for client in self.selected_clients:
            try:
                self.client_data.append(client.send_to_server())
            except Exception as e:
                self.logger.error(
                    f"Failed to receive data from client {client.id}: {e}"
                )

    def initialize_clients(self) -> None:
        """
        Instantiates client objects based on the strategy configuration.

        Dynamically resolves the client class name based on the server's module.
        """
        module_name = self.__module__
        class_name = self.__class__.__name__ + "_Client"
        try:
            client_class = getattr(
                __import__(module_name, fromlist=[class_name]), class_name
            )
        except (ImportError, AttributeError):
            # Fallback to base Client class if specialized one isn't found
            client_class = Client
        self.clients = [
            client_class(configs=self.configs, id=cid, times=self.times)
            for cid in range(self.num_clients)
        ]

    def save_results(self) -> None:
        """Saves both server-side and all individual client-side metrics."""
        super().save_results()
        for client in self.clients:
            client.save_results()

    def save_models(self, save_type: str, verbose: bool = True) -> None:
        """Saves the current model(s) to disk.

        Args:
            save_type: The save criteria ('last' or 'best').
            verbose: Whether to log the save operation.

        Raises:
            ValueError: If an unsupported save_type is provided.
        """
        if save_type not in ["last", "best"]:
            raise ValueError("save_type must be 'last' or 'best'")

        should_save = True

        if save_type == "best":
            metric_key = (
                "personal_avg_test_loss"
                if self.save_local_model
                else "global_avg_test_loss"
            )

            if metric_key not in self.metrics or not self.metrics[metric_key]:
                should_save = False
            else:
                metric_values = self.metrics[metric_key]
                if metric_values[-1] != min(metric_values):
                    should_save = False

        if not should_save:
            return

        postfix = save_type

        # Save server/global model
        if not self.exclude_server_model_processes:
            self.save_model(
                model=self.model,
                path=self.model_path,
                name=self.name,
                postfix=postfix,
                verbose=self.logger,
            )

        # If local models are not to be saved, return after server model (if saved)
        if not self.save_local_model:
            return

        # Save client/local models
        for client in self.clients:
            client.save_model(
                model=client.model,
                path=client.model_path,
                name=client.name,
                postfix=postfix,
                verbose=client.logger,
            )

    def early_stopping(self) -> bool:
        """
        Determines if the training process should stop early based on patience.

        Checks the trend of test loss (local or global depending on config).

        Returns:
            True if early stopping criteria are met, False otherwise.
        """
        metric = (
            self.metrics["personal_avg_test_loss"]
            if self.save_local_model
            else self.metrics["global_avg_test_loss"]
        )

        if not self.patience or len(metric) < self.patience:
            return False

        # Find the minimum loss so far
        best_so_far = min(metric)

        # Check if the minimum value is in the last `self.patience` rounds
        if best_so_far not in metric[-self.patience :]:
            self.logger.info("Early stopping activated.")
            return True

        return False

    def evaluate_generalization_loss(self, dataset_type: str) -> None:
        """
        Evaluates the global model's performance across all clients.

        Args:
            dataset_type: The dataset split to evaluate ('train', 'test', or 'valid').
        """
        if self.parallel:
            futures = []
            for client in self.clients:
                device = "cuda" if self.num_gpus > 0 else "cpu"
                num_gpus = 1 / self.num_workers if self.num_gpus > 0 else 0

                future = ray.remote(num_gpus=num_gpus)(
                    lambda cl, model, criterion, dtype, dev: (
                        lambda: (
                            model.to(dev),
                            model.eval(),
                            float(
                                np.mean(
                                    [
                                        criterion(
                                            model(x.float().to(dev)), y.float().to(dev)
                                        ).item()
                                        for x, y in getattr(cl, f"load_{dtype}_data")()
                                    ]
                                )
                            ),
                        )
                    )()[2]
                ).remote(
                    client,
                    self.model,
                    client.loss,
                    dataset_type,
                    device,
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
                for client in self.clients
            ]
        metric_name = f"global_avg_{dataset_type}_loss"
        self.metrics[metric_name].append(float(np.mean(losses)))
        self.logger.info(
            f"Generalization {dataset_type.capitalize()} Loss: "
            f"{self.metrics[metric_name][-1]:.4f}"
        )

    def evaluate_personalization_loss(self, dataset_type: str) -> None:
        """
        Evaluates each client's personalized model performance locally.

        Args:
            dataset_type: The dataset split to evaluate ('train', 'test', or 'valid').
        """
        if self.parallel:
            futures = []
            for client in self.clients:
                device = "cuda" if self.num_gpus > 0 else "cpu"
                num_gpus = 1 / self.num_workers if self.num_gpus > 0 else 0

                future = ray.remote(num_gpus=num_gpus)(
                    lambda cl, dtype, dev: (
                        setattr(cl, "device", dev),
                        float(getattr(cl, f"get_{dtype}_loss")()),
                    )[1]
                ).remote(
                    client,
                    dataset_type,
                    device,
                )
                futures.append(future)
            losses = ray.get(futures)
        else:
            losses = [
                float(getattr(client, f"get_{dataset_type}_loss")())
                for client in self.clients
            ]
        metric_name = f"personal_avg_{dataset_type}_loss"
        self.metrics[metric_name].append(float(np.mean(losses)))
        self.logger.info(
            f"Personalization {dataset_type.capitalize()} Loss: "
            f"{self.metrics[metric_name][-1]:.4f}"
        )

    def train_clients(self) -> None:
        """
        Orchestrates the training of selected clients in the current round.

        Handles both parallel execution via Ray and serial execution on the local machine.
        Results are collected and local client states are updated automatically.
        """
        if self.parallel:
            i = 0
            futures = []
            idle_workers = deque(range(self.num_workers))
            job_map = {}
            client_packages = {}

            while i < len(self.selected_clients) or len(futures) > 0:
                while i < len(self.selected_clients) and len(idle_workers) > 0:
                    worker_id = idle_workers.popleft()
                    client = self.selected_clients[i]

                    # Parallelize the `train()` method using Ray
                    # We use a lambda to ensure the remote worker has access to the client instance
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

                        # Update client model & optimizer parameters from the remote worker's package
                        client.update_model_params(
                            old=client.model, new=client_package["model"]
                        )
                        client.update_optimizer_params(
                            old=client.optimizer, new=client_package["optimizer"]
                        )
                        client.metrics["train_time"].append(
                            client_package["train_time"]
                        )
                        client.train_samples = client_package["train_samples"]

        else:
            # Serial execution for debugging or small-scale runs
            for client in self.selected_clients:
                client.train()

    def fix_results(self) -> None:
        """Pads results for the server and all clients to ensure consistent metric lengths."""
        super().fix_results(default=self.default_value)
        for client in self.clients:
            client.fix_results(default=self.default_value)

    def calculate_aggregation_weights(self) -> None:
        """Computes weight for each client based on their contribution score (e.g., sample size)."""
        ts = [client["score"] for client in self.client_data]
        self.weights = torch.tensor(ts) / sum(ts)

    def aggregate_models(self) -> None:
        """
        Aggregates client models into the global server model.

        Supports both standard averaging and difference-based aggregation (Federated Averaging).
        """
        if self.return_diff:
            for client, weight in zip(self.client_data, self.weights):
                for global_param, local_param in zip(
                    self.model.parameters(), client["model"].parameters()
                ):
                    global_param.data.sub_(local_param.data, alpha=weight)
        else:
            self.model = self.reset_model(self.model)
            for client, weight in zip(self.client_data, self.weights):
                for global_param, local_param in zip(
                    self.model.parameters(), client["model"].parameters()
                ):
                    global_param.data.add_(local_param.data, alpha=weight)

    def get_model_info(self) -> None:
        """
        Generates architectural summaries for the server and client models.

        Computes input/output shapes and visualizes the network structure.
        """
        if not self.exclude_server_model_processes:
            dl = self.clients[0].load_train_data()
            self.summarize_model(dataloader=dl)
            del dl
            gc.collect()
        if self.save_local_model:
            for client in self.clients:
                dl = client.load_train_data()
                client.summarize_model(dataloader=dl)
                del dl
                gc.collect()

    def post_process(self) -> None:
        """
        Executes cleanup and final saving operations after training completion.

        Saves the final model, exports results, closes loggers, and shuts down Ray.
        """
        self.logger.info("")
        self.logger.info("-" * 50)
        self.save_models(save_type="last")
        self.save_results()

        # close all client loggers + server logger
        for c in self.clients:
            try:
                c.close_logger()
            except Exception:
                pass
        try:
            self.close_logger()
        except Exception:
            pass

        # shutdown ray gracefully
        try:
            import ray

            ray.shutdown()
        except Exception:
            pass

    def pre_train_clients(self) -> None:
        """Hook for executing operations before client training begins each round."""

    def train(self) -> None:
        """
        Main federated learning training loop.

        Iterates through the specified number of communications rounds,
        orchestrating the end-to-end FL process.
        """
        for i in range(self.iterations):
            round_start_time = time.time()
            self.current_iter = i

            self.logger.info("")
            self.logger.info(
                f"-------------Round number: {str(self.current_iter).zfill(4)}-------------"
            )

            self.select_clients()
            self.send_to_clients()

            # Optional Pre-aggregation Evaluation
            if self.current_iter % self.eval_gap == 0:
                for dataset_type in ["train", "test"]:
                    if dataset_type == "train" and self.skip_eval_train:
                        continue
                    if self.save_local_model:
                        # Personalization loss evaluation
                        self.evaluate_personalization_loss(dataset_type)

            self.pre_train_clients()
            self.train_clients()
            self.receive_from_clients()
            self.calculate_aggregation_weights()
            self.aggregate_models()

            # Post-aggregation Evaluation
            if self.current_iter % self.eval_gap == 0:
                for dataset_type in ["train", "test"]:
                    if dataset_type == "train" and self.skip_eval_train:
                        continue
                    if not self.exclude_server_model_processes:
                        # Generalization loss evaluation
                        self.evaluate_generalization_loss(dataset_type)

            self.save_models(save_type="best")
            round_duration = time.time() - round_start_time
            self.metrics["time_per_iter"].append(round_duration)
            self.logger.info(f"Time cost: {round_duration:.4f}s")

            self.fix_results()
            if self.early_stopping():
                break

        self.post_process()


class Client(SharedMethods):
    """
    Represents a local node in the federated learning network.

    Handles local data loading, model training, and communication with the server.
    Inherits from SharedMethods for utility functions.
    """

    def __init__(self, configs: Namespace, id: int, times: int) -> None:
        """
        Initializes the Client instance with provided configurations.

        Args:
            configs: Global configuration namespace.
            id: Unique identifier for the client.
            times: Experiment iteration/repetition index.
        """
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

        self.metrics = {
            "train_time": [],
            "train_loss": [],
            "test_loss": [],
            "send_mb": [],
            "lr": [],
        }

    def initialize_private_info(self) -> None:
        """Loads client-specific metadata and data paths from the info file."""
        with open(self.path_info, "r") as f:
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
        """Initializes the data scaler based on client-specific statistics."""
        self.scaler = getattr(__import__("scalers"), self.scaler)(self.stats)

    def variables_to_be_sent(self) -> Dict[str, Any]:
        """
        Prepares the local parameters/differentials to be sent to the server.

        Supports sending full models or model differentials (weights - snapshot).

        Returns:
            A dictionary containing the local model and sample count.
        """
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
        return {"model": model, "score": self.train_samples}

    def send_to_server(self) -> Dict[str, Any]:
        """
        Initiates the data transfer to the server and tracks communication volume.

        Returns:
            The local data package (model, score, etc.).
        """
        to_be_sent = self.variables_to_be_sent()
        total_size = sum(self.get_size(value) for value in to_be_sent.values())
        self.metrics["send_mb"].append(total_size)
        return to_be_sent

    def load_train_data(
        self, sample_ratio: float = 1.0, shuffle: bool = True
    ) -> DataLoader:
        """
        Loads the client's local training dataset.

        Args:
            sample_ratio: Fraction of the local dataset to use.
            shuffle: Whether to shuffle the training data.

        Returns:
            A PyTorch DataLoader for training.
        """
        trainloader = self.load_data(
            file=self.train_file,
            sample_ratio=sample_ratio,
            shuffle=shuffle,
            scaler=self.scaler,
            batch_size=self.batch_size,
        )
        self.train_samples = len(trainloader.dataset)
        return trainloader

    def load_test_data(
        self, sample_ratio: float = 1.0, shuffle: bool = False
    ) -> DataLoader:
        """
        Loads the client's local test dataset.

        Args:
            sample_ratio: Fraction of the local dataset to use.
            shuffle: Whether to shuffle the test data.

        Returns:
            A PyTorch DataLoader for testing.
        """
        testloader = self.load_data(
            file=self.test_file,
            sample_ratio=sample_ratio,
            shuffle=shuffle,
            scaler=self.scaler,
            batch_size=self.batch_size,
        )
        self.test_samples = len(testloader.dataset)
        return testloader

    def train(self) -> Optional[Dict[str, Any]]:
        """
        Executes local training for the specified number of epochs.

        Returns:
            In parallel mode, returns a dictionary containing the updated model,
            optimizer, and metrics. Returns None in serial mode.
        """
        train_loader = self.load_train_data()
        start_time = time.time()
        for _ in range(self.epochs):
            self.train_one_epoch(
                model=self.model,
                dataloader=train_loader,
                optimizer=self.optimizer,
                criterion=self.loss,
                scheduler=self.scheduler,
                device=self.device,
            )
        train_time = time.time() - start_time
        if self.parallel:
            return {
                "id": self.id,
                "model": self.model,
                "optimizer": self.optimizer,
                "train_time": train_time,
                "train_samples": self.train_samples,
            }
        self.metrics["train_time"].append(train_time)
        self.metrics["lr"].append(self.scheduler.get_last_lr()[0])
        return None

    def get_train_loss(self) -> float:
        """
        Calculates the average training loss on the local dataset.

        Returns:
            The mean scalar loss across all training batches.
        """
        losses = self.calculate_loss(
            model=self.model,
            dataloader=self.load_train_data(),
            criterion=self.loss,
            device=self.device,
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
        )
        losses = np.mean(losses)
        self.metrics["test_loss"].append(losses)
        return losses

    def receive_from_server(self, data):
        if self.return_diff:
            self.snapshot = copy.deepcopy(data["model"]).to("cpu")
        self.update_model_params(old=self.model, new=data["model"])
