import torch

from .base import Loss


class MdAPE(Loss):
    """
    Median absolute percentage error
    """

    def forward(self, input, target):
        return torch.median(
            input=torch.abs(
                self._percentage_error(
                    input=input,
                    target=target,
                )
            )
        )
