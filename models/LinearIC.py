import torch
import torch.nn as nn


class LinearIC(nn.Module):
    """
    Source: https://github.com/cure-lab/LTSF-Linear/blob/main/models/Linear.py
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.input_len
        self.pred_len = configs.output_len
        self.channels = configs.input_channels

        self.Linear = nn.ModuleList()
        for i in range(self.channels):
            self.Linear.append(nn.Linear(self.seq_len, self.pred_len))

    def forward(self, x):
        # x: [B, seq_len, in_channels]
        output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(
            x.device
        )
        for i in range(self.channels):
            output[:, :, i] = self.Linear[i](x[:, :, i])
        return output  # [B, pred_len, D]
