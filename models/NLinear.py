import torch
import torch.nn as nn


class NLinear(nn.Module):
    """
    Source: https://github.com/cure-lab/LTSF-Linear/blob/main/models/NLinear.py
    """

    def __init__(self, configs):
        super(NLinear, self).__init__()
        self.seq_len = configs.input_len
        self.pred_len = configs.output_len

        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        self.Linear.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
        )

    def forward(self, x):
        # x: [B, seq_len, in_channels]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x  # [B, pred_len, out_channels]
