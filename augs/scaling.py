import torch


class scaling:
    """Scale each sample feature by a Gaussian factor."""

    def __init__(self, sigma: float = 0.5) -> None:
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        factor = 1.0 + self.sigma * torch.randn(
            size=(x.size(0), 1, x.size(2)),
            device=x.device,
            dtype=x.dtype,
        )
        return x * factor
