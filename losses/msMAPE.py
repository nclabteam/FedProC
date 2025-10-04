import torch

from .base import Loss


class msMAPE(Loss):
    """
    Modified Symmetric Mean Absolute Percentage Error
    """

    def forward(self, input, target):
        return torch.mean(
            input=self._symmetric_absolute_percentage_error(
                input=input,
                target=target,
            )
        )
