import torch


class jitter:
    """Add zero-mean Gaussian noise."""

    def __init__(self, sigma: float = 0.3) -> None:
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.sigma * torch.randn_like(input=x)
