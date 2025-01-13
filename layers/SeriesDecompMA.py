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
