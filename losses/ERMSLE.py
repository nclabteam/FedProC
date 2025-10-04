import torch

from .RMSLE import RMSLE


class ERMSLE(RMSLE):
    """
    Exponential Root Mean Squared Log Error
    ERMSLE = exp(RMSLE)
    """

    def forward(self, input, target):
        rmsle = super().forward(input=input, target=target)
        return torch.exp(rmsle)
