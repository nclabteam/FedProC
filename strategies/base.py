import copy
import inspect
import logging
import os
import platform
import subprocess
import sys
from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import ray
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from schedulers import SCHEDULER_MODES
from utils.seed import SetSeed

os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")


class SharedMethods:
    """Collection of small reusable utilities used by Server and Client classes."""

    default_value = 9_999_999.0
    scheduler_mode = "iteration"
    checkpoint_format = "fedproc_state_dict_v1"
    checkpoint_format_version = 1

    @staticmethod
    def mean_models(
        models: List[Dict[str, torch.Tensor]],
        weights: Optional[Union[List[float], torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return a coordinate-wise model mean."""
        if not models:
            raise ValueError("at least one model is required")
        if weights is None:
            weights = torch.ones(len(models), dtype=torch.float64)
        else:
            weights = torch.as_tensor(list(weights), dtype=torch.float64)
            if weights.numel() != len(models):
                raise ValueError("one weight per model is required")
        total = weights.sum()
        if not torch.isfinite(total) or total <= 0:
            raise ValueError("model weights must have a positive finite sum")
        weights = weights / total

        averaged = type(models[0])()
        for name in models[0]:
            values = torch.stack([model[name] for model in models])
            if not values.is_floating_point() and not values.is_complex():
                averaged[name] = values[0].clone()
                continue
            dtype = (
                torch.float32
                if values.dtype in (torch.float16, torch.bfloat16)
                else values.dtype
            )
            values = values.to(dtype=dtype)
            averaged[name] = torch.tensordot(
                weights.to(device=values.device, dtype=dtype),
                values,
                dims=([0], [0]),
            ).to(models[0][name].dtype)
        return averaged

    @staticmethod
    def load_data(
        file: str,
        sample_ratio: float = 1.0,
        batch_size: int = 32,
        shuffle: bool = False,
        scaler: Any = None,
        seed: Optional[int] = None,
        indices: Optional[List[int]] = None,
    ) -> DataLoader:
        assert 0 <= sample_ratio <= 1, "sample_ratio must be between 0 and 1"
        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)

        with np.load(file) as data:
            x = data["x"]
            y = data["y"]
            x_mark = torch.as_tensor(np.asarray(data["x_mark"], dtype=np.float32))
            y_mark = torch.as_tensor(np.asarray(data["y_mark"], dtype=np.float32))
        x = torch.as_tensor(np.asarray(scaler.transform(x), dtype=np.float32))
        y = torch.as_tensor(np.asarray(scaler.transform(y), dtype=np.float32))
        dataset = TensorDataset(x, y, x_mark, y_mark)

        if indices is not None:
            # Caller-supplied deterministic subset (e.g. strided/non-overlapping window
            # selection) takes precedence over the random sample_ratio subset below.
            dataset = Subset(dataset, indices)
        elif sample_ratio < 1.0:
            subset_size = int(len(dataset) * sample_ratio)
            rand_indices = torch.randperm(len(dataset), generator=generator)[
                :subset_size
            ]
            dataset = Subset(dataset, rand_indices)

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
        )

    @staticmethod
    def calculate_loss(
        model: torch.nn.Module,
        dataloader: DataLoader,
        criterion: Callable,
        device: Union[str, torch.device] = "cpu",
        offload_after: bool = True,
    ) -> List[float]:
        losses = []
        model.to(device)
        model.eval()
        with torch.no_grad():
            for batch_x, batch_y, x_mark, y_mark in dataloader:
                batch_x = batch_x.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                batch_y = batch_y.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                x_mark = x_mark.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                y_mark = y_mark.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                outputs = model(batch_x, x_mark=x_mark, y_mark=y_mark)
                loss = criterion(outputs, batch_y)
                losses.append(loss.item())
        if offload_after:
            model.to("cpu")
        return losses

    @staticmethod
    def save_model(
        model: torch.nn.Module,
        path: str,
        name: str,
        postfix: str = "",
        extention: str = "pt",
        configs: Optional[Namespace] = None,
        metadata: Optional[Dict[str, Any]] = None,
        verbose: Optional[logging.Logger] = None,
    ) -> None:
        save_path = os.path.join(
            path,
            f"{'_'.join([name.lower().strip(), postfix])}.{extention}",
        )
        checkpoint = SharedMethods.build_checkpoint(
            model=model,
            configs=configs,
            metadata=metadata,
        )
        torch.save(obj=checkpoint, f=save_path)
        if verbose is not None:
            verbose.info(f"Model saved to {save_path}")

    @staticmethod
    def _to_serializable(value: Any) -> Any:
        if isinstance(value, Namespace):
            return SharedMethods._to_serializable(value=vars(value))
        if isinstance(value, dict):
            return {
                key: SharedMethods._to_serializable(value=item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [SharedMethods._to_serializable(value=item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    @staticmethod
    def _checkpoint_metadata(
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        metadata = {
            "torch_version": str(torch.__version__),
            "python_version": platform.python_version(),
            "git_commit": SharedMethods._get_git_commit(),
        }
        if extra_metadata:
            metadata.update(SharedMethods._to_serializable(value=extra_metadata))
        return metadata

    @staticmethod
    def _get_git_commit() -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip() or None
        except Exception:
            return None

    @staticmethod
    def build_checkpoint(
        model: torch.nn.Module,
        configs: Optional[Namespace] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        config_dict = (
            SharedMethods._to_serializable(value=vars(configs)) if configs else {}
        )
        model_name = config_dict.get("model", model.__class__.__name__)
        state_dict = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        return {
            "format": SharedMethods.checkpoint_format,
            "format_version": SharedMethods.checkpoint_format_version,
            "model_name": model_name,
            "config": config_dict,
            "state_dict": state_dict,
            "metadata": SharedMethods._checkpoint_metadata(extra_metadata=metadata),
        }

    @staticmethod
    def load_checkpoint_model(
        checkpoint_path: str,
        device: Union[str, torch.device] = "cpu",
        allow_unsafe_legacy: bool = False,
        verbose: Optional[logging.Logger] = None,
    ) -> torch.nn.Module:
        def log_warning(message: str) -> None:
            if verbose is not None:
                verbose.warning(message)

        try:
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except Exception as error:
            if not allow_unsafe_legacy:
                raise ValueError(
                    f"Checkpoint {checkpoint_path} is not in the safe FedProC state-dict "
                    "format. Re-run with unsafe legacy loading explicitly enabled."
                ) from error
            log_warning(
                f"Unsafe legacy checkpoint load enabled for {checkpoint_path}. "
                "This may execute arbitrary code during deserialization."
            )
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )

        if (
            isinstance(payload, dict)
            and payload.get("format") == SharedMethods.checkpoint_format
        ):
            model_name = payload["model_name"]
            config = Namespace(**payload.get("config", {}))
            model_cls = SharedMethods._get_objective_function(
                func_type="models", func_name=model_name
            )
            model = model_cls(configs=config)
            model.load_state_dict(payload["state_dict"])
            return model.to(device)

        if isinstance(payload, torch.nn.Module):
            if not allow_unsafe_legacy:
                raise ValueError(
                    f"Checkpoint {checkpoint_path} uses the legacy full-object format. "
                    "Enable unsafe legacy loading explicitly to open it."
                )
            log_warning(
                f"Unsafe legacy checkpoint load enabled for {checkpoint_path}. "
                "This may execute arbitrary code during deserialization."
            )
            return payload.to(device)

        raise ValueError(
            f"Checkpoint {checkpoint_path} is not in a supported FedProC format."
        )

    @staticmethod
    def get_size(obj: Any) -> float:
        return SharedMethods._get_size_bytes(obj=obj) / (1024**2)

    @staticmethod
    def _get_size_bytes(obj: Any) -> float:
        if isinstance(obj, torch.Tensor):
            return obj.element_size() * obj.nelement()
        if isinstance(obj, torch.nn.Module):
            total_size = sum(
                param.element_size() * param.nelement() for param in obj.parameters()
            )
            total_size += sum(
                buffer.element_size() * buffer.nelement() for buffer in obj.buffers()
            )
            return total_size
        if isinstance(obj, DataLoader):
            total_size = sum(
                sum(
                    item.element_size() * item.nelement()
                    for item in data
                    if isinstance(item, torch.Tensor)
                )
                for data in obj.dataset
            )
            return total_size
        if isinstance(obj, TensorDataset):
            total_size = sum(
                tensor.element_size() * tensor.nelement() for tensor in obj.tensors
            )
            return total_size
        if isinstance(obj, dict):
            return sum(
                SharedMethods._get_size_bytes(obj=value) for value in obj.values()
            )
        if isinstance(obj, (list, tuple)):
            return sum(SharedMethods._get_size_bytes(obj=item) for item in obj)
        if isinstance(obj, np.ndarray):
            return obj.nbytes
        if isinstance(obj, np.generic):
            return obj.itemsize
        if isinstance(obj, bool):
            return 1
        if isinstance(obj, (int, float)):
            # wire width of a scalar, not the Python object header: sys.getsizeof
            # bills a float at 24 bytes and an int at 28, inflating any payload
            # that carries scalars alongside tensors
            return 4
        return sys.getsizeof(obj)

    @staticmethod
    def _derive_seed(base_seed: Optional[int], *parts: int) -> Optional[int]:
        if base_seed is None:
            return None
        seed = int(base_seed) & 0xFFFFFFFF
        for part in parts:
            seed = (seed * 1664525 + int(part) + 1013904223) & 0xFFFFFFFF
        return seed

    @staticmethod
    def _set_worker_seed(seed: Optional[int]) -> None:
        if seed is None:
            return
        SetSeed.set_all(seed=seed, verbose=False)

    @staticmethod
    def train_one_epoch(
        model: torch.nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
        scheduler: Optional[Any],
        device: Union[str, torch.device],
        offload_after: bool = True,
    ) -> None:
        model.to(device)
        SharedMethods._move_optimizer_state_to_param_devices(optimizer=optimizer)
        model.train()
        for batch_x, batch_y, x_mark, y_mark in dataloader:
            optimizer.zero_grad(set_to_none=True)
            batch_x = batch_x.to(device=device, dtype=torch.float32, non_blocking=True)
            batch_y = batch_y.to(device=device, dtype=torch.float32, non_blocking=True)
            x_mark = x_mark.to(device=device, dtype=torch.float32, non_blocking=True)
            y_mark = y_mark.to(device=device, dtype=torch.float32, non_blocking=True)
            outputs = model(batch_x, x_mark=x_mark, y_mark=y_mark)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            SharedMethods.step_scheduler_batch(
                scheduler=scheduler,
                batch_data=batch_x,
            )
        if offload_after:
            model.to("cpu")
        SharedMethods.step_scheduler_epoch(scheduler=scheduler)

    @staticmethod
    def step_scheduler_batch(
        scheduler: Optional[Any],
        batch_data: torch.Tensor,
    ) -> None:
        """Advance a batch-level scheduler."""
        if (
            scheduler is None
            or getattr(scheduler, "scheduler_mode", "iteration") != "batch"
        ):
            return
        if hasattr(scheduler, "set_batch_data"):
            scheduler.set_batch_data(data=batch_data)
        scheduler.step()

    @staticmethod
    def step_scheduler_epoch(scheduler: Optional[Any]) -> None:
        """Advance an epoch- or iteration-level scheduler."""
        if (
            scheduler is None
            or getattr(scheduler, "scheduler_mode", "iteration") == "batch"
        ):
            return
        scheduler.step()

    @staticmethod
    def _move_optimizer_state_to_param_devices(
        optimizer: torch.optim.Optimizer,
    ) -> None:
        def move(value: Any, device: torch.device) -> Any:
            if isinstance(value, torch.Tensor):
                return value.to(device=device)
            if isinstance(value, dict):
                return {key: move(item, device) for key, item in value.items()}
            if isinstance(value, list):
                return [move(item, device) for item in value]
            if isinstance(value, tuple):
                return tuple(move(item, device) for item in value)
            return value

        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter in optimizer.state:
                    optimizer.state[parameter] = move(
                        optimizer.state[parameter],
                        parameter.device,
                    )

    @staticmethod
    def _optimizer_state_to_cpu(optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        state_dict = copy.deepcopy(optimizer.state_dict())

        def to_cpu(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            if isinstance(value, dict):
                return {key: to_cpu(item) for key, item in value.items()}
            if isinstance(value, list):
                return [to_cpu(item) for item in value]
            if isinstance(value, tuple):
                return tuple(to_cpu(item) for item in value)
            return value

        return to_cpu(state_dict)

    @staticmethod
    def _get_objective_function(func_type: str, func_name: str) -> Callable:
        module = __import__(func_type, fromlist=[func_name])
        func = getattr(module, func_name)
        if inspect.ismodule(func) and hasattr(func, func_name):
            return getattr(func, func_name)
        return func

    def initialize_loss(self) -> None:
        obj = self._get_objective_function(func_type="losses", func_name=self.loss)
        self.loss = obj()

    def initialize_model(self) -> None:
        obj = self._get_objective_function(func_type="models", func_name=self.model)
        self.model = obj(configs=self.configs)

    def initialize_optimizer(self) -> None:
        obj = self._get_objective_function(
            func_type="optimizers", func_name=self.optimizer
        )
        self.optimizer = obj(params=self.model.parameters(), configs=self.configs)
        self._scheduler_base_lrs = [
            float(group["lr"]) for group in self.optimizer.param_groups
        ]

    @staticmethod
    def _scheduler_configs(
        configs: Namespace,
        steps_per_epoch: int,
        epochs: Optional[int] = None,
    ) -> Namespace:
        """Return scheduler-local configs with the correct step horizon."""
        mode = getattr(configs, "scheduler_mode", "iteration")
        if mode not in SCHEDULER_MODES:
            raise ValueError(f"unknown scheduler mode: {mode}")
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive")
        epochs = int(configs.epochs if epochs is None else epochs)
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        total_steps = {
            "batch": epochs * steps_per_epoch,
            "epoch": epochs,
            "iteration": int(configs.max_epochs),
        }[mode]
        scheduler_configs = copy.copy(configs)
        scheduler_configs.max_epochs = total_steps
        scheduler_configs.steps_per_epoch = steps_per_epoch
        scheduler_configs.total_steps = total_steps
        return scheduler_configs

    def initialize_scheduler(
        self,
        steps_per_epoch: int = 1,
        epochs: Optional[int] = None,
    ) -> None:
        if (
            self.scheduler_mode == "iteration"
            and getattr(getattr(self, "scheduler", None), "optimizer", None)
            is self.optimizer
        ):
            return
        obj = self._get_objective_function(
            func_type="schedulers", func_name=self.configs.scheduler
        )
        if not hasattr(self, "_scheduler_base_lrs"):
            self._scheduler_base_lrs = [
                float(group["lr"]) for group in self.optimizer.param_groups
            ]
        if len(self._scheduler_base_lrs) != len(self.optimizer.param_groups):
            raise ValueError("scheduler needs one base LR per optimizer group")
        for group, learning_rate in zip(
            self.optimizer.param_groups,
            self._scheduler_base_lrs,
        ):
            group["lr"] = learning_rate
            group["initial_lr"] = learning_rate
        self.scheduler = obj(
            optimizer=self.optimizer,
            configs=self._scheduler_configs(
                configs=self.configs,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
            ),
        )
        self.scheduler.scheduler_mode = self.scheduler_mode
        self.init_scheduler_state = copy.deepcopy(self.scheduler.state_dict())

    @staticmethod
    def restore_scheduler(
        scheduler: Any,
        optimizer: torch.optim.Optimizer,
        state: Dict[str, Any],
        mode: str,
    ) -> None:
        """Restore scheduler progress and its optimizer learning rates."""
        scheduler.load_state_dict(state)
        scheduler.scheduler_mode = mode
        rates = scheduler.get_last_lr()
        if len(rates) != len(optimizer.param_groups):
            raise ValueError("scheduler state has the wrong number of learning rates")
        for group, learning_rate in zip(optimizer.param_groups, rates):
            group["lr"] = learning_rate

    def summarize_model(self, dataloader: DataLoader) -> None:
        import torchinfo

        sample = next(iter(dataloader))
        input_size = tuple(sample[0].shape)
        original_device = next(self.model.parameters()).device
        result = torchinfo.summary(
            self.model,
            input_size=input_size,
            device=self.device,
            verbose=0,
            col_names=["output_size", "num_params", "mult_adds"],
        )
        self.model.to(original_device)
        try:
            print(result)
        except UnicodeEncodeError:
            sys.stdout.buffer.write(str(result).encode("utf-8"))
            sys.stdout.buffer.write(b"\n")
            sys.stdout.buffer.flush()

        name = self.name.lower().strip().replace(" ", "_")
        path = os.path.join(self.models_info_path, f"{name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(result))

    def make_logger(self, name: str, path: str) -> None:
        log_path = os.path.join(path, f"{name.lower().strip()}.log")
        logger_name = f"{name}_{self.times}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers():
            for h in list(self.logger.handlers):
                try:
                    h.flush()
                    h.close()
                except Exception:
                    pass
            self.logger.handlers.clear()
        self.logger.propagate = False
        file_handler = logging.FileHandler(log_path)
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter(f"%(asctime)s ~ {name} ~ %(message)s")
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
        self.logger.info(f"Logger created at {log_path}")

    def close_logger(self) -> None:
        if hasattr(self, "logger"):
            for h in list(self.logger.handlers):
                try:
                    h.flush()
                    h.close()
                except Exception:
                    pass
            self.logger.handlers.clear()

    def set_configs(self, configs: Namespace, **kwargs: Any) -> None:
        if isinstance(configs, Namespace):
            for key, value in vars(configs).items():
                setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.configs = configs

    def mkdir(self) -> None:
        self.save_path = os.path.join(self.save_path, str(self.times))
        self.model_path = os.path.join(self.save_path, "models")
        self.log_path = os.path.join(self.save_path, "logs")
        self.result_path = os.path.join(self.save_path, "results")
        self.models_info_path = os.path.join(self.save_path, "models_info")
        for dir_path in [
            self.save_path,
            self.model_path,
            self.log_path,
            self.result_path,
            self.models_info_path,
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
    seed = client._loader_seed(dataset_type)
    client._set_worker_seed(seed)
    if mode == "generalization":
        if model is None or criterion is None:
            raise ValueError("model and criterion are required for generalization mode")
        model = model.to(device)
        dataloader = getattr(client, f"load_{dataset_type}_data")()
        losses = []
        model.eval()
        with torch.no_grad():
            for batch_x, batch_y, x_mark, y_mark in dataloader:
                batch_x = batch_x.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                batch_y = batch_y.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                x_mark = x_mark.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                y_mark = y_mark.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                outputs = model(batch_x, x_mark=x_mark, y_mark=y_mark)
                loss = criterion(outputs, batch_y)
                losses.append(float(loss.item()))
        return float(np.mean(losses))

    elif mode == "personalization":
        client.device = device
        return float(getattr(client, f"get_{dataset_type}_loss")())

    else:
        raise ValueError(f"Unsupported mode for ray_compute_client_loss: {mode}")
