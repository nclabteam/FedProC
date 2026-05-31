import torch


class jitter:
    def __init__(self, sigma=0.3):
        self.sigma = sigma

    def __call__(self, x):
        return x + torch.normal(mean=0.0, std=self.sigma, size=x.shape, device=x.device)
