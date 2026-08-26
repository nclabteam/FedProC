import torch
from torch import Tensor

from .base import Loss


class msMAPE(Loss):
    """Compute modified symmetric mean absolute percentage error."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return torch.mean(
            self._modified_symmetric_absolute_percentage_error(
                input=input,
                target=target,
            )
        )
