import torch
import torch.nn as nn


class SeriesDecompMA(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.ma = MovingAverage(kernel_size, stride=stride)

    def forward(self, x):
        moving_average = self.ma(x)
        res = x - moving_average
        return res, moving_average


class MovingAverage(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(MovingAverage, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecompMultiMA(nn.Module):
    """
    Series decomposition block with multiple kernel sizes (for FEDformer).
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = [MovingAverage(kernel, stride=1) for kernel in kernel_size]
        self.layer = nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean = []
        for func in self.moving_avg:
            ma = func(x)
            moving_mean.append(ma.unsqueeze(-1))
        moving_mean = torch.cat(moving_mean, dim=-1)
        moving_mean = torch.sum(
            moving_mean * nn.Softmax(-1)(self.layer(x.unsqueeze(-1))), dim=-1
        )
        res = x - moving_mean
        return res, moving_mean
