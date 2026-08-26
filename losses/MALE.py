import torch
from torch import Tensor

from .base import Loss


class MALE(Loss):
    """Compute mean absolute log error for positive values."""

    generic_eval = False

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return torch.mean(torch.abs(self._log_error(input=input, target=target)))
