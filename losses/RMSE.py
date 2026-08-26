import math

import torch
from torch import Tensor

from .base import Loss


class RMSE(Loss):
    """Compute root mean squared error."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return torch.linalg.vector_norm(input - target) / math.sqrt(input.numel())
