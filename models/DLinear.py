import torch
import torch.nn as nn

from layers import SeriesDecompMA

optional = {
    "moving_avg": 25,
    "stride": 1,
}


def args_update(parser):
    parser.add_argument("--moving_avg", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)


class DLinear(nn.Module):
    """
    Paper: https://arxiv.org/abs/2205.13504
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.input_len
        self.pred_len = configs.output_len
        self.decompsition = SeriesDecompMA(
            kernel_size=configs.moving_avg, stride=configs.stride
        )

        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

        self.Linear_Seasonal.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
        )
        self.Linear_Trend.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
        )

    def forward(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(
            0, 2, 1
        )
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # [B, L, D]
