import torch
from torch import Tensor

from .base import Loss


class RSquared(Loss):
    """Compute the coefficient of determination for evaluation."""

    eval_only = True

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        residual = torch.sum(torch.square(target - input))
        total = torch.sum(torch.square(target - torch.mean(target)))
        score = 1 - residual / total.clamp_min(torch.finfo(total.dtype).eps)
        return torch.where(total > 0, score, (residual == 0).to(input.dtype))
