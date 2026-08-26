from collections.abc import Sequence

import torch
from torch import Tensor, nn


class MQC(nn.Module):
    """Measure multi-quantile change between aligned consecutive forecasts."""

    context_only = True
    DEFAULT_QUANTILES = (
        0.005,
        0.025,
        0.05,
        0.1,
        0.15,
        0.2,
        0.5,
        0.8,
        0.85,
        0.9,
        0.95,
        0.975,
        0.995,
    )

    def __init__(self, quantiles: Sequence[float] | Tensor | None = None) -> None:
        super().__init__()
        values = torch.as_tensor(
            self.DEFAULT_QUANTILES if quantiles is None else quantiles,
            dtype=torch.get_default_dtype(),
        ).flatten()
        if values.numel() == 0 or torch.any((values <= 0) | (values >= 1)):
            raise ValueError("quantiles must contain values strictly between 0 and 1")
        self.register_buffer("quantiles", values)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if input.shape != target.shape or input.ndim == 0:
            raise ValueError("current and previous forecasts must have the same shape")
        if input.shape[-1] != self.quantiles.numel():
            raise ValueError("the final forecast dimension must match quantiles")

        # Paper MQC definition: apply pinball change against the previous forecast.
        quantiles = self.quantiles.to(input)
        change = target - input
        return torch.where(
            change >= 0,
            quantiles * change,
            (1 - quantiles) * -change,
        ).mean()
