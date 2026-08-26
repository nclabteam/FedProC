import torch
from torch import Tensor

from .base import Loss


class MdAPE(Loss):
    """Compute median absolute percentage error."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return torch.median(
            torch.abs(self._percentage_error(input=input, target=target))
        )
