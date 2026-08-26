import torch
from torch import Tensor

from .base import Loss


class MAPE(Loss):
    """Compute mean absolute percentage error."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return torch.mean(torch.abs(self._percentage_error(input=input, target=target)))
