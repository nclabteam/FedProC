import torch

from .base import Loss


class MAPE(Loss):
    """
    Mean Absolute Percentage Error
    """

    def forward(self, input, target):
        return (
            torch.mean(torch.abs(self.divide_no_nan(a=target - input, b=target))) * 100
        )
