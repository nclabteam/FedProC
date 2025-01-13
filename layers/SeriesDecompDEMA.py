import torch
from torch import nn


class SeriesDecompDEMA(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, alpha, beta, learnable=False, device="cuda"):
        super().__init__()
        self.ma = DoubleExponentialMovingAverage(
            alpha=alpha, beta=beta, learnable=learnable, device=device
        )

    def forward(self, x):
        moving_average = self.ma(x)
        res = x - moving_average
        return res, moving_average


class DoubleExponentialMovingAverage(nn.Module):
    """
    Double Exponential Moving Average (DEMA) block to highlight the trend of time series
    """

    def __init__(self, alpha, beta, learnable=False, device="cuda"):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        if learnable:
            self.alpha = nn.Parameter(
                torch.tensor(self.alpha, dtype=torch.float32, device=device)
            )
            self.beta = nn.Parameter(
                torch.tensor(self.beta, dtype=torch.float32, device=device)
            )
        assert 0 <= self.alpha <= 1, "DEMA alpha should be in [0, 1]"
        assert 0 <= self.beta <= 1, "DEMA beta should be in [0, 1]"

    def forward(self, x):
        s_prev = x[:, 0, :]
        b = x[:, 1, :] - s_prev
        res = [s_prev.unsqueeze(1)]
        for t in range(1, x.shape[1]):
            xt = x[:, t, :]
            s = self.alpha * xt + (1 - self.alpha) * (s_prev + b)
            b = self.beta * (s - s_prev) + (1 - self.beta) * b
            s_prev = s
            res.append(s.unsqueeze(1))
        return torch.cat(res, dim=1)
