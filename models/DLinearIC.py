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


class DLinearIC(nn.Module):
    """
    Paper: https://arxiv.org/abs/2205.13504
    Source: https://github.com/cure-lab/LTSF-Linear/blob/main/models/DLinear.py
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.input_len
        self.pred_len = configs.output_len
        self.channels = configs.input_channels

        self.decompsition = SeriesDecompMA(
            kernel_size=configs.moving_avg, stride=configs.stride
        )

        self.Linear_Seasonal = nn.ModuleList()
        self.Linear_Trend = nn.ModuleList()
        for _ in range(self.channels):
            self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
            self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))

    def forward(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        seasonal_output = torch.zeros(
            [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
            dtype=seasonal_init.dtype,
        ).to(seasonal_init.device)
        trend_output = torch.zeros(
            [trend_init.size(0), trend_init.size(1), self.pred_len],
            dtype=trend_init.dtype,
        ).to(trend_init.device)
        for i in range(self.channels):
            seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
            trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # [B, L, D]
