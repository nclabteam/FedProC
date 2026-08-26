import math

import torch
from torch import Tensor

from .base import Loss


class RMSSE(Loss):
    """Compute RMSSE using forecast targets and separate in-sample history."""

    context_only = True

    def __init__(self, seasonal_period: int = 1) -> None:
        super().__init__()
        if seasonal_period <= 0:
            raise ValueError("seasonal_period must be positive")
        self.seasonal_period = seasonal_period

    def forward(self, input: Tensor, target: Tensor, insample: Tensor) -> Tensor:
        if input.shape != target.shape:
            raise ValueError("prediction and target must have the same shape")
        if insample.ndim < 2 or insample.shape[-2] <= self.seasonal_period:
            raise ValueError("in-sample history must exceed the seasonal period")

        # Forecast Evaluation RMSSE definition: scale OOS RMSE by seasonal-naive IS RMSE.
        forecast_rmse = torch.linalg.vector_norm(input - target) / math.sqrt(
            input.numel()
        )
        naive_error = (
            insample[..., self.seasonal_period :, :]
            - insample[..., : -self.seasonal_period, :]
        )
        naive_rmse = torch.linalg.vector_norm(naive_error) / math.sqrt(
            naive_error.numel()
        )
        return forecast_rmse / naive_rmse.clamp_min(torch.finfo(naive_rmse.dtype).eps)
