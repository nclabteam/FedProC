import torch

from .MALE import MALE


class EMALE(MALE):
    """
    Exponential Mean Absolute Log Error
    EMALE = exp(MALE)
    """

    def forward(self, input, target):
        male = super().forward(input=input, target=target)
        return torch.exp(male)
