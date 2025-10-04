import torch

from .base import Loss


class RMdSPE(Loss):
    """
    Root Median Square Percentage Error
    """

    def forward(self, input, target):
        return torch.sqrt(
            input=torch.median(
                input=torch.square(
                    self._percentage_error(
                        input=input,
                        target=target,
                    )
                )
            )
        )
