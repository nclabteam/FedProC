import torch

from .base import Loss


class RSquared(Loss):
    """
    Coefficient of determination (R^2) score
    """

    def forward(self, input, target):
        target_mean = torch.mean(target)
        ss_tot = torch.sum((target - target_mean) ** 2)
        ss_res = torch.sum((target - input) ** 2)
        r2_score = 1 - ss_res / ss_tot
        return r2_score
