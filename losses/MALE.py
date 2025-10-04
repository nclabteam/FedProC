import torch

from .base import Loss


class MALE(Loss):
    """
    Mean Absolute Log Error
    https://towardsdatascience.com/mean-absolute-log-error-male-a-better-relative-performance-metric-a8fd17bc5f75/
    """

    def forward(self, input, target):
        return torch.mean(torch.abs(self._log_error(x=input, y=target)))
