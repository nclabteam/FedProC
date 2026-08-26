import torch
from torch import Tensor

from .RMSLE import RMSLE


class ERMSLE(RMSLE):
    """Compute exponentiated root mean squared log error."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return torch.exp(super().forward(input=input, target=target))
