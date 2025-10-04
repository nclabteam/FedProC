import torch

from .base import Loss


class sMdAPE(Loss):
    """
    Symmetric Median Absolute Percentage Error
    """

    def forward(self, input, target):
        return torch.median(
            input=self._symmetric_abosulute_percentage_error(
                input=input,
                target=target,
            )
        )
