import torch
import torch.nn as nn


class Linear(nn.Module):
    """
    Source: https://github.com/cure-lab/LTSF-Linear/blob/main/models/Linear.py
    """

    def __init__(self, configs):
        super().__init__()
        self.Linear = nn.Linear(configs.input_len, configs.output_len)

    def forward(self, x):
        # x: [batch_size, input_len, in_channels]
        x = x.permute(0, 2, 1)
        # x: [batch_size, in_channels, input_len]
        x = self.Linear(x)
        # x: [batch_size, in_channels, output_len]
        x = x.permute(0, 2, 1)
        # x: [batch_size, output_len, in_channels]
        return x
