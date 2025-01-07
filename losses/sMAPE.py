import torch

from .base import Loss


class sMAPE(Loss):
    """
    Symmetric Mean Absolute Percentage Error
    """

    def forward(self, input, target):
        return (
            torch.mean(
                torch.abs(
                    self.divide_no_nan(
                        a=torch.abs(target - input),
                        b=(torch.abs(target) + torch.abs(input)) / 2,
                    )
                )
            )
            * 100
        )
