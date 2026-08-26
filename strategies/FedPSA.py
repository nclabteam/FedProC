"""FedPSA behavioral-staleness aggregation."""

import math
from argparse import Namespace
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from .aFL import aFL
from .tFL import tFL_Client


class FedPSAShared:
    """FedPSA operations shared by the server and reusable workers."""

    @staticmethod
    def projection_matrix(
        num_params: int,
        sketch_dim: int,
        seed: int,
    ) -> torch.Tensor:
        if num_params <= 0 or sketch_dim <= 0:
            raise ValueError("num_params and sketch_dim must be positive")
        generator = torch.Generator().manual_seed(seed)
        return torch.randn(
            sketch_dim,
            num_params,
            generator=generator,
        ) / math.sqrt(sketch_dim)

    @staticmethod
    def calibration_batch(
        batch_size: int,
        input_len: int,
        input_channels: int,
        output_len: int,
        output_channels: int,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dimensions = (
            batch_size,
            input_len,
            input_channels,
            output_len,
            output_channels,
        )
        if any(dimension <= 0 for dimension in dimensions):
            raise ValueError("calibration dimensions must be positive")
        generator = torch.Generator().manual_seed(seed)
        return (
            torch.randn(
                batch_size,
                input_len,
                input_channels,
                generator=generator,
            ),
            torch.randn(
                batch_size,
                output_len,
                output_channels,
                generator=generator,
            ),
        )

    @staticmethod
    def sensitivity_sketch(
        model: torch.nn.Module,
        calibration_x: torch.Tensor,
        calibration_y: torch.Tensor,
        projection: torch.Tensor,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        if not parameters:
            raise ValueError("FedPSA requires trainable model parameters")

        device = parameters[0].device
        was_training = model.training
        model.eval()
        try:
            output = model(calibration_x.to(device=device))
            loss = criterion(output, calibration_y.to(device=device))
            gradients = torch.autograd.grad(
                outputs=loss,
                inputs=parameters,
                allow_unused=True,
            )
        finally:
            model.train(mode=was_training)

        flat_parameters = torch.cat(
            [parameter.detach().flatten() for parameter in parameters]
        )
        flat_gradients = torch.cat(
            [
                (
                    gradient.detach().flatten()
                    if gradient is not None
                    else torch.zeros_like(parameter).flatten()
                )
                for parameter, gradient in zip(parameters, gradients)
            ]
        )
        if projection.ndim != 2 or projection.shape[1] != flat_parameters.numel():
            raise ValueError(
                "projection width must match the trainable parameter count"
            )

        # Paper sensitivity: s = |gθ - 1/2 Fθ²|, with F approximated by g².
        sensitivity = (
            flat_gradients * flat_parameters
            - 0.5 * flat_gradients.square() * flat_parameters.square()
        ).abs()
        # Paper sketch: s_tilde = R s.
        return (
            projection.to(device=device, dtype=sensitivity.dtype)
            .mv(sensitivity)
            .detach()
            .cpu()
        )

    @staticmethod
    def similarity_weights(
        sketches: Sequence[torch.Tensor],
        global_sketch: torch.Tensor,
        temperature: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not sketches:
            raise ValueError("at least one client sketch is required")
        matrix = torch.stack(list(sketches)).to(dtype=global_sketch.dtype)
        reference = global_sketch.unsqueeze(dim=0)
        # Paper behavioral similarity: kappa_i = cos(s_tilde_i, s_tilde_g).
        similarities = F.cosine_similarity(matrix, reference, dim=1, eps=1e-8)
        if temperature is None:
            weights = torch.full_like(similarities, 1.0 / len(sketches))
        else:
            if not math.isfinite(temperature) or temperature <= 0:
                raise ValueError("temperature must be positive and finite")
            # Paper aggregation weight: Weight_i = softmax(kappa_i / Temp).
            weights = F.softmax(similarities / temperature, dim=0)
        return weights, similarities


class FedPSA(FedPSAShared, aFL):
    """Parameter-sensitivity asynchronous FL (Lu et al., 2026)."""

    optional = {
        "buffer_size": 5,
        "queue_len": 50,
        "sketch_dim": 16,
        "gamma": 5.0,
        "delta": 0.5,
        "calib_size": 32,
    }
    compulsory = {"return_diff": True}
    _projection_seed = 0
    _calibration_seed = 1

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        if int(self.queue_len) <= 0:
            raise ValueError("queue_len must be positive")
        if self.gamma < 0 or self.delta <= 0:
            raise ValueError("gamma must be nonnegative and delta must be positive")

        num_params = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        self._projection = self.projection_matrix(
            num_params=num_params,
            sketch_dim=int(self.sketch_dim),
            seed=self._projection_seed,
        )
        self._calibration_x, self._calibration_y = self.calibration_batch(
            batch_size=int(self.calib_size),
            input_len=int(self.input_len),
            input_channels=int(self.input_channels),
            output_len=int(self.output_len),
            output_channels=int(self.output_channels),
            seed=self._calibration_seed,
        )
        self._criterion = self._get_objective_function(
            func_type="losses",
            func_name=configs.loss,
        )()
        self._thermometer: deque[float] = deque(maxlen=int(self.queue_len))
        self._initial_magnitude: float | None = None

    def aggregate_client_updates(
        self,
        packages: Mapping[int, dict[str, Any]],
    ) -> None:
        if not packages:
            raise ValueError("at least one client update is required")

        model_diffs = []
        client_sketches = []
        for package in packages.values():
            model_diff = package["model_params_diff"]
            if not model_diff:
                raise ValueError("FedPSA requires model parameter differences")
            magnitude = float(
                torch.stack(
                    [value.float().square().sum() for value in model_diff.values()]
                ).sum()
            )
            if not math.isfinite(magnitude):
                raise ValueError("client update magnitude must be finite")
            self._thermometer.append(magnitude)
            if (
                self._initial_magnitude is None
                and len(self._thermometer) == self._thermometer.maxlen
            ):
                self._initial_magnitude = sum(self._thermometer) / len(
                    self._thermometer
                )
            model_diffs.append(model_diff)
            client_sketches.append(package["_psa_s_tilde"])

        current_magnitude = sum(self._thermometer) / len(self._thermometer)
        # Paper thermometer: Temp = (M_cur / M_0) gamma + delta.
        temperature = (
            None
            if self._initial_magnitude is None
            else current_magnitude
            / max(self._initial_magnitude, torch.finfo(torch.float32).eps)
            * self.gamma
            + self.delta
        )
        global_sketch = self.sensitivity_sketch(
            model=self.model,
            calibration_x=self._calibration_x,
            calibration_y=self._calibration_y,
            projection=self._projection,
            criterion=self._criterion,
        )
        weights, similarities = self.similarity_weights(
            sketches=client_sketches,
            global_sketch=global_sketch,
            temperature=temperature,
        )
        mean_diff = self.mean_models(
            models=model_diffs,
            weights=weights,
        )
        # Paper update, using model_params_diff = w_dispatched - w_local.
        self._commit_global(
            new_params=OrderedDict(
                (
                    name,
                    parameter
                    - mean_diff[name].to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    ),
                )
                for name, parameter in self.public_model_params.items()
            )
        )
        self.logger.info(
            "FedPSA aggregation: temp=%s, kappa=%s",
            "uniform" if temperature is None else f"{temperature:.4f}",
            " ".join(f"{value:.3f}" for value in similarities.tolist()),
        )


class FedPSA_Client(FedPSAShared, tFL_Client):
    """Reusable FedPSA worker with immutable shared sketch inputs."""

    return_diff = True
    return_diff_score = False

    def __init__(self, configs: Namespace, times: int, device: str) -> None:
        super().__init__(configs=configs, times=times, device=device)
        num_params = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        self._projection = self.projection_matrix(
            num_params=num_params,
            sketch_dim=int(self.sketch_dim),
            seed=FedPSA._projection_seed,
        )
        self._calibration_x, self._calibration_y = self.calibration_batch(
            batch_size=int(self.calib_size),
            input_len=int(self.input_len),
            input_channels=int(self.input_channels),
            output_len=int(self.output_len),
            output_channels=int(self.output_channels),
            seed=FedPSA._calibration_seed,
        )

    def package(self) -> dict[str, Any]:
        result = super().package()
        result["_psa_s_tilde"] = self.sensitivity_sketch(
            model=self.model,
            calibration_x=self._calibration_x,
            calibration_y=self._calibration_y,
            projection=self._projection,
            criterion=self.loss,
        )
        result["regular_model_params"] = {}
        result["__wire__"] = ("model_params_diff", "_psa_s_tilde")
        return result
