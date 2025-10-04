import torch

from .base import Loss


class RMSLE(Loss):
    """
    Root Mean Squared Log Error
    https://towardsdatascience.com/mean-absolute-log-error-male-a-better-relative-performance-metric-a8fd17bc5f75/
    """

    def forward(self, input, target):
        return torch.sqrt(torch.mean(self._log_error(x=input, y=target) ** 2))
