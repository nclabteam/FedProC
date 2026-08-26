import torch
from torch import Tensor

from .base import Loss


class RMdSPE(Loss):
    """Compute root median squared percentage error."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return torch.median(
            torch.abs(self._percentage_error(input=input, target=target))
        )
