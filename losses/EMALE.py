import torch
from torch import Tensor

from .MALE import MALE


class EMALE(MALE):
    """Compute exponentiated mean absolute log error."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return torch.exp(super().forward(input=input, target=target))
