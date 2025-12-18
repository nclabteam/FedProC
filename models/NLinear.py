import torch.nn as nn


class NLinear(nn.Module):
    """
    Source: https://github.com/cure-lab/LTSF-Linear/blob/main/models/NLinear.py
    """

    def __init__(self, configs):
        super().__init__()
        self.Linear = nn.Linear(configs.input_len, configs.output_len)

    def forward(self, x):
        # x: [batch_size, input_len, in_channels]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        # x: [batch_size, in_channels, input_len]
        x = x.permute(0, 2, 1)
        # x: [batch_size, in_channels, input_len]
        x = self.Linear(x)
        # x: [batch_size, in_channels, output_len]
        x = x.permute(0, 2, 1)
        # x: [batch_size, output_len, in_channels]
        x = x + seq_last
        # x: [batch_size, output_len, in_channels]
        return x
