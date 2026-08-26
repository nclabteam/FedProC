import torch
from torch import Tensor

from .base import Loss


class sMAPE(Loss):
    """Compute symmetric mean absolute percentage error."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return torch.mean(
            self._symmetric_absolute_percentage_error(
                input=input,
                target=target,
            )
        )
