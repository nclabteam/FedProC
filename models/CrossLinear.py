import math

import torch
import torch.nn as nn

optional = {
    "d_model": 512,
    "d_ff": 2048,
    "patch_len": 4,
    "alpha": 1,
    "beta": 0.5,
}


def args_update(parser):
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--patch_len", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)


class CrossLinear(nn.Module):
    """
    Paper: https://arxiv.org/pdf/2505.23116
    Source: https://github.com/mumiao2000/CrossLinear/blob/main/models/CrossLinear.py
    """

    def __init__(self, configs):
        super(CrossLinear, self).__init__()
        self.EPS = 1e-5
        patch_len = configs.patch_len
        patch_num = math.ceil(configs.input_len / patch_len)

        # embedding
        self.alpha = nn.Parameter(torch.ones([1]) * configs.alpha)
        self.beta = nn.Parameter(torch.ones([1]) * configs.beta)
        self.correlation_embedding = nn.Conv1d(
            in_channels=configs.output_channels,
            out_channels=configs.input_channels,
            kernel_size=3,
            padding="same",
        )
        self.value_embedding = Patch_Embedding(
            seq_len=configs.input_len,
            patch_num=patch_num,
            patch_len=patch_len,
            d_model=configs.d_model,
            d_ff=configs.d_ff,
            variate_num=configs.input_channels,
        )
        self.pos_embedding = nn.Parameter(
            torch.randn(1, configs.input_channels, patch_num, configs.d_model)
        )

        # head
        self.head = De_Patch_Embedding(
            pred_len=configs.output_len,
            patch_num=patch_num,
            d_model=configs.d_model,
            d_ff=configs.d_ff,
            variate_num=configs.input_channels,
        )

    def forward(self, x_enc):
        x_enc = x_enc.permute(0, 2, 1)
        # normalization
        x_obj = x_enc
        mean = torch.mean(x_obj, dim=-1, keepdim=True)
        std = torch.std(x_obj, dim=-1, keepdim=True)
        x_enc = (x_enc - torch.mean(x_enc, dim=-1, keepdim=True)) / (
            torch.std(x_enc, dim=-1, keepdim=True) + self.EPS
        )
        # embedding
        x_obj = x_enc
        x_obj = self.alpha * x_obj + (1 - self.alpha) * self.correlation_embedding(
            x_enc
        )
        x_obj = (
            self.beta * self.value_embedding(x_obj)
            + (1 - self.beta) * self.pos_embedding
        )
        # head
        y_out = self.head(x_obj)
        # de-normalization
        y_out = y_out * std + mean
        y_out = y_out.permute(0, 2, 1)
        return y_out


class Patch_Embedding(nn.Module):
    def __init__(self, seq_len, patch_num, patch_len, d_model, d_ff, variate_num):
        super(Patch_Embedding, self).__init__()
        self.pad_num = patch_num * patch_len - seq_len
        self.patch_len = patch_len
        self.linear = nn.Sequential(
            nn.LayerNorm([variate_num, patch_num, patch_len]),
            nn.Linear(patch_len, d_ff),
            nn.LayerNorm([variate_num, patch_num, d_ff]),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm([variate_num, patch_num, d_model]),
            nn.ReLU(),
        )

    def forward(self, x):
        x = nn.functional.pad(x, (0, self.pad_num))
        x = x.unfold(2, self.patch_len, self.patch_len)
        x = self.linear(x)
        return x


class De_Patch_Embedding(nn.Module):
    def __init__(self, pred_len, patch_num, d_model, d_ff, variate_num):
        super(De_Patch_Embedding, self).__init__()
        self.linear = nn.Sequential(
            nn.Flatten(2),
            nn.Linear(patch_num * d_model, d_ff),
            nn.LayerNorm([variate_num, d_ff]),
            nn.ReLU(),
            nn.Linear(d_ff, pred_len),
        )

    def forward(self, x):
        x = self.linear(x)
        return x
