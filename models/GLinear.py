import torch.nn as nn


class GLinear(nn.Module):
    """
    Paper: https://arxiv.org/abs/2501.01087
    Source: https://github.com/t-rizvi/GLinear/blob/main/models/WIthout_Normalization/DNGLinear.py
    """

    def __init__(self, configs):
        super().__init__()

        self.seq_len = configs.input_len
        self.pred_len = configs.output_len

        self.Linear = nn.Linear(self.seq_len, self.seq_len)
        self.GeLU = nn.GELU()
        self.Hidden1 = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.Linear(x)
        x = self.GeLU(x)
        x = self.Hidden1(x)
        x = x.permute(0, 2, 1)
        return x  # [Batch, Output length, Channel]
