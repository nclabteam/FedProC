import torch

from .base import Loss


class RMSPE(Loss):
    """
    Root Mean Squared Percentage Error
    """

    def forward(self, input, target):
        return torch.sqrt(
            input=torch.mean(
                input=torch.square(
                    input=self._percentage_error(
                        input=input,
                        target=target,
                    )
                )
            )
        )
