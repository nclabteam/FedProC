import math

import torch
from torch import Tensor

from .base import Loss


class RMSLE(Loss):
    """Compute root mean squared log error for positive values."""

    generic_eval = False

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        error = self._log_error(input=input, target=target)
        return torch.linalg.vector_norm(error) / math.sqrt(error.numel())
