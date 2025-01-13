import torch.nn as nn

from .DLinear import DLinear, args_update, optional
from .NLinear import NLinear


class DNGLinear(nn.Module):
    """
    Paper: https://arxiv.org/abs/2501.01087
    Source: https://github.com/t-rizvi/GLinear/blob/main/models/WIthout_Normalization/DNGLinear.py
    """

    def __init__(self, configs):
        super().__init__()

        self.seq_len = configs.input_len
        self.pred_len = configs.output_len
        self.input_channels = configs.input_channels

        self.nlinear = NLinear(configs)
        self.dlinear = DLinear(configs)
        self.LinrLyr = nn.Linear(self.seq_len, self.pred_len)

        self.Linear = nn.Linear(self.seq_len, self.seq_len)
        self.GeLU = nn.GELU()
        self.Hidden1 = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        x1 = self.nlinear(x)
        x2 = self.dlinear(x)
        x3 = x.permute(0, 2, 1)

        x3 = self.Linear(x3)
        x3 = self.GeLU(x3)
        x3 = self.Hidden1(x3)

        x3 = x3.permute(0, 2, 1)
        x4 = self.LinrLyr(x.permute(0, 2, 1)).permute(0, 2, 1)

        x = (x1 + x2 + x3 + x4) / 4
        return x  # [Batch, Output length, Channel]
