import torch
from torch import nn


class SeriesDecompEMA(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, alpha, learnable=False, device="cuda"):
        super().__init__()
        self.ma = ExponentialMovingAverage(
            alpha=alpha, learnable=learnable, device=device
        )

    def forward(self, x):
        moving_average = self.ma(x)
        res = x - moving_average
        return res, moving_average


class ExponentialMovingAverage(nn.Module):
    """
    Exponential Moving Average (EMA) block to highlight the trend of time series
    """

    def __init__(self, alpha, learnable=False, device="cuda"):
        super().__init__()
        self.alpha = alpha
        if learnable:
            self.alpha = nn.Parameter(
                torch.tensor(self.alpha, dtype=torch.float32, device=device)
            )
        assert 0 <= self.alpha <= 1, "EMA alpha should be in [0, 1]"

    # Optimized implementation with O(1) time complexity
    def forward(self, x):
        # x: [Batch, Input, Channel]
        _, t, _ = x.shape
        powers = torch.flip(torch.arange(t, dtype=torch.double), dims=(0,))
        weights = torch.pow((1 - self.alpha), powers).to("cuda")
        divisor = weights.clone()
        weights[1:] = weights[1:] * self.alpha
        weights = weights.reshape(1, t, 1)
        divisor = divisor.reshape(1, t, 1)
        x = torch.cumsum(x * weights, dim=1)
        x = torch.div(x, divisor)
        return x.to(torch.float32)
