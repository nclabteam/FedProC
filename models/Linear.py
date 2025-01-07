import torch
import torch.nn as nn


class Linear(nn.Module):
    """
    Source: https://github.com/cure-lab/LTSF-Linear/blob/main/models/Linear.py
    """

    def __init__(self, configs):
        super(Linear, self).__init__()
        self.seq_len = configs.input_len
        self.pred_len = configs.output_len

        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        self.Linear.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
        )

    def forward(self, x):
        # x: [B, seq_len, in_channels]
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x  # [B, pred_len, D]
