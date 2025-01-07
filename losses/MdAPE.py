import torch

from .base import Loss


class MdAPE(Loss):
    """
    Median absolute percentage error
    """

    def forward(self, input, target):
        return (
            torch.median(torch.abs(self.divide_no_nan(a=target - input, b=target)))
            * 100
        )
