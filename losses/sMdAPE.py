import torch
from torch import Tensor

from .base import Loss


class sMdAPE(Loss):
    """Compute symmetric median absolute percentage error."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return torch.median(
            self._symmetric_absolute_percentage_error(
                input=input,
                target=target,
            )
        )
