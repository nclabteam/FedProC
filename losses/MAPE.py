import torch

from .base import Loss


class MAPE(Loss):
    """
    Mean Absolute Percentage Error
    """

    def forward(self, input, target):
        return torch.mean(
            input=torch.abs(
                input=self._percentage_error(
                    input=input,
                    target=target,
                )
            )
        )
