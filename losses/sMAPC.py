import torch
from torch import Tensor

from .base import Loss


class sMAPC(Loss):
    """Measure symmetric percentage change between aligned forecasts."""

    context_only = True

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if input.shape != target.shape:
            raise ValueError("current and previous forecasts must have the same shape")

        # Paper sMAPC definition: compare forecasts for the same periods at adjacent origins.
        return 200 * torch.mean(
            self.divide_no_nan(
                a=torch.abs(input - target),
                b=torch.abs(input) + torch.abs(target),
            )
        )
