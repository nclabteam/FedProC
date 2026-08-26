import math

import torch
from torch import Tensor

from .base import Loss


class RMSPE(Loss):
    """Compute root mean squared percentage error."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        error = self._percentage_error(input=input, target=target)
        return torch.linalg.vector_norm(error) / math.sqrt(error.numel())
