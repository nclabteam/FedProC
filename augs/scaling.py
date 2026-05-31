import torch


class scaling:
    def __init__(self, sigma=0.5):
        self.sigma = sigma

    def __call__(self, x):
        # x: [B, T, D]
        factor = torch.normal(
            mean=1.0, std=self.sigma, size=(x.size(0), 1, x.size(2)), device=x.device
        )
        return x * factor
