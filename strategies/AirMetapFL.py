import argparse
import math
from collections import OrderedDict
from typing import Any

import torch

from .mFL import mFL, mFL_Client


class AirMetapFLShared:
    """AirMeta-pFL update compression and over-the-air estimation."""

    @staticmethod
    def flatten_params(params: Any) -> torch.Tensor:
        return torch.cat(
            [value.detach().reshape(-1).float() for value in params.values()]
        )

    @staticmethod
    def unflatten_params(vector: torch.Tensor, template: Any) -> OrderedDict:
        result = OrderedDict()
        offset = 0
        for name, value in template.items():
            result[name] = (
                vector[offset : offset + value.numel()]
                .reshape_as(value)
                .to(value.dtype)
            )
            offset += value.numel()
        if offset != vector.numel():
            raise ValueError("AirMeta-pFL update size does not match the model")
        return result

    @staticmethod
    def top_k(vector: torch.Tensor, ratio: float) -> torch.Tensor:
        if not 0 < ratio <= 1:
            raise ValueError("AirMeta-pFL sparsity must be in (0, 1]")
        count = min(vector.numel(), max(1, int(ratio * vector.numel())))
        if count == vector.numel():
            return vector.clone()
        result = torch.zeros_like(vector)
        indices = torch.topk(vector.abs(), count, sorted=False).indices
        result[indices] = vector[indices]
        return result

    @classmethod
    def compress_with_memory(
        cls,
        update: torch.Tensor,
        memory: torch.Tensor,
        ratio: float,
    ) -> Any:
        corrected = update + memory
        compressed = cls.top_k(vector=corrected, ratio=ratio)
        return compressed, corrected - compressed

    @staticmethod
    def _sparse_fourier_estimate(
        received: torch.Tensor,
        indices: torch.Tensor,
        dimensions: int,
        support_size: int,
        steps: int,
    ) -> torch.Tensor:
        # ponytail: the paper leaves estimator E open; use dependency-free IHT
        # until experiment parity specifically requires the reported OAMP estimator.
        estimate = torch.zeros(dimensions, dtype=torch.float32)
        for _ in range(max(1, steps)):
            residual = received - torch.fft.fft(estimate, norm="ortho")[indices]
            spectrum = torch.zeros(dimensions, dtype=received.dtype)
            spectrum[indices] = residual
            estimate.add_(torch.fft.ifft(spectrum, norm="ortho").real.float())
            if support_size < dimensions:
                keep = torch.topk(estimate.abs(), support_size, sorted=False).indices
                sparse = torch.zeros_like(estimate)
                sparse[keep] = estimate[keep]
                estimate = sparse
        return estimate

    @classmethod
    def aggregate_over_air(
        cls,
        updates: Any,
        compression_ratio: float,
        sparsity: float,
        learning_rate: float,
        power: float,
        noise_std: float,
        channel_mean: float,
        estimator_steps: int,
        seed: int,
        channel_gains: Any = None,
    ) -> torch.Tensor:
        if not updates:
            raise ValueError("AirMeta-pFL requires at least one update")
        if not 0 < compression_ratio <= 1:
            raise ValueError("AirMeta-pFL compression_ratio must be in (0, 1]")
        if learning_rate <= 0 or power <= 0 or noise_std < 0 or channel_mean <= 0:
            raise ValueError("AirMeta-pFL channel and learning parameters are invalid")

        vectors = [update.detach().cpu().float().reshape(-1) for update in updates]
        dimensions = vectors[0].numel()
        if any(vector.numel() != dimensions for vector in vectors):
            raise ValueError("AirMeta-pFL updates must have equal dimensions")
        client_count = len(vectors)
        vectors = torch.stack(vectors)
        measurements = max(1, math.ceil(compression_ratio * dimensions))
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(dimensions, generator=generator)[:measurements]
        spectra = torch.fft.fft(vectors, dim=1, norm="ortho")[:, indices]
        max_energy = float(spectra.abs().square().sum(dim=1).max())
        if max_energy == 0:
            return torch.zeros(dimensions)
        rho = power * measurements * learning_rate**2 / max_energy

        if channel_gains is None:
            sigma = channel_mean * math.sqrt(2.0 / math.pi)
            uniform = torch.rand(client_count, generator=generator).clamp_min(1e-12)
            gains = sigma * torch.sqrt(-2.0 * torch.log(uniform))
        else:
            gains = torch.as_tensor(channel_gains, dtype=torch.float32)
            if gains.numel() != client_count:
                raise ValueError("AirMeta-pFL needs one channel gain per update")

        received = (gains.reshape(-1, 1) * spectra).sum(dim=0)
        received.mul_(math.sqrt(rho) / learning_rate)
        if noise_std:
            noise = torch.complex(
                torch.randn(measurements, generator=generator),
                torch.randn(measurements, generator=generator),
            ) * (noise_std / math.sqrt(2.0))
            received = received + noise

        support_size = min(
            dimensions,
            max(1, math.ceil(sparsity * dimensions) * client_count),
        )
        estimate = cls._sparse_fourier_estimate(
            received=received,
            indices=indices,
            dimensions=dimensions,
            support_size=support_size,
            steps=estimator_steps,
        )
        return estimate * (
            learning_rate / (channel_mean * math.sqrt(rho) * client_count)
        )


class AirMetapFL(AirMetapFLShared, mFL):
    """AirMeta-pFL with error feedback and analog partial-DFT aggregation."""

    compulsory = {"return_diff": True}
    optional = {
        "alpha": 0.4,
        "delta": 1e-3,
        "sparsity": 0.04,
        "hf": True,
        "compression_ratio": 0.4,
        "air_power": 1.0,
        "air_noise_std": 0.0,
        "air_channel_mean": 1.0,
        "air_estimator_steps": 20,
    }

    @classmethod
    def args_update(cls, parser: Any) -> Any:
        parser.add_argument("--alpha", type=float, default=None)
        parser.add_argument("--delta", type=float, default=None)
        parser.add_argument("--sparsity", type=float, default=None)
        parser.add_argument("--hf", action=argparse.BooleanOptionalAction, default=None)
        parser.add_argument("--compression_ratio", type=float, default=None)
        parser.add_argument("--air_power", type=float, default=None)
        parser.add_argument("--air_noise_std", type=float, default=None)
        parser.add_argument("--air_channel_mean", type=float, default=None)
        parser.add_argument("--air_estimator_steps", type=int, default=None)
        return parser

    def package(self, client_id: int) -> dict:
        package = super().package(client_id=client_id)
        package["__wire__"] = ("regular_model_params",)
        return package

    def aggregate_client_updates(self, packages: Any) -> None:
        template = self.public_model_params
        updates = [
            self.flatten_params(params=package["air_update"])
            for package in packages.values()
        ]
        estimate = self.aggregate_over_air(
            updates=updates,
            compression_ratio=self.compression_ratio,
            sparsity=self.sparsity,
            learning_rate=self.learning_rate,
            power=self.air_power,
            noise_std=self.air_noise_std,
            channel_mean=self.air_channel_mean,
            estimator_steps=self.air_estimator_steps,
            seed=int(self.seed or 0) + self.times + self.current_iter,
        )
        current = self.flatten_params(params=template)
        self._commit_global(
            new_params=self.unflatten_params(
                vector=current - estimate,
                template=template,
            )
        )

    def _compute_send_mb(self, packages: Any) -> tuple:
        dimensions = sum(value.numel() for value in self.public_model_params.values())
        measurements = max(1, math.ceil(self.compression_ratio * dimensions))
        symbol_mb = measurements * 8 / (1024**2)
        incumbents = sum(not is_new for is_new in self.is_new.values())
        downlink = self.get_size(obj=self.public_model_params) * incumbents
        return {cid: symbol_mb for cid in packages}, downlink


class AirMetapFL_Client(AirMetapFLShared, mFL_Client):
    """AirMeta-pFL meta worker with persistent sparsification memory."""

    return_diff = True
    return_diff_score = False

    def __init__(self, configs: Any, times: Any, device: Any) -> None:
        super().__init__(configs=configs, times=times, device=device)
        self._memory = None

    def _inner_learning_rate(self) -> float:
        return self.alpha

    def _outer_learning_rate(self) -> float:
        return self.learning_rate

    def set_parameters(self, package: dict) -> None:
        super().set_parameters(package=package)
        memory = package["personal_model_params"].get("air_memory")
        self._memory = None if memory is None else memory.detach().clone()

    def package(self) -> dict:
        package = super().package()
        delta = self.flatten_params(params=package.pop("model_params_diff"))
        if self._memory is None:
            self._memory = torch.zeros_like(delta)
        if self._memory.numel() != delta.numel():
            raise ValueError("AirMeta-pFL memory size does not match the model")
        compressed, self._memory = self.compress_with_memory(
            update=delta,
            memory=self._memory,
            ratio=self.sparsity,
        )
        package["air_update"] = self.unflatten_params(
            vector=compressed,
            template=package["regular_model_params"],
        )
        package["personal_model_params"]["air_memory"] = self._memory
        package["__wire__"] = ()
        return package
