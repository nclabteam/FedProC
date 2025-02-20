import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import RevIN

optional = {
    "embed_size": 256,
    "hidden_size": 512,
    "dropout": 0,
    "band_width": 96,
    "scale": 0.02,
    "sparsity_threshold": 0.01,
}


def args_update(parser):
    parser.add_argument("--embed_size", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--band_width", type=int, default=None)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--sparsity_threshold", type=float, default=None)


class TexFilter(nn.Module):
    """
    Paper: https://arxiv.org/abs/2411.01623
    Source: https://github.com/aikunyi/FilterNet/blob/main/models/TexFilter.py
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.input_len
        self.pred_len = configs.output_len
        self.embed_size = configs.embed_size
        self.hidden_size = configs.hidden_size
        self.dropout = configs.dropout
        self.channels = configs.input_channels
        self.band_width = configs.band_width
        self.scale = configs.scale
        self.sparsity_threshold = configs.sparsity_threshold

        self.revin_layer = RevIN(self.channels, affine=True)
        self.embedding = nn.Linear(self.seq_len, self.embed_size)
        self.token = nn.Conv1d(
            in_channels=self.seq_len, out_channels=self.embed_size, kernel_size=(1,)
        )

        self.w = nn.Parameter(self.scale * torch.randn(2, self.embed_size))
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.embed_size))

        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.embed_size),
        )

        self.output = nn.Linear(self.embed_size, self.pred_len)
        self.layernorm = nn.LayerNorm(self.embed_size)
        self.layernorm1 = nn.LayerNorm(self.embed_size)
        self.dropout = nn.Dropout(self.dropout)

    def tokenEmbed(self, x):
        x = self.token(x)
        return x

    def texfilter(self, x):
        B, N, _ = x.shape
        o1_real = torch.zeros([B, N // 2 + 1, self.embed_size], device=x.device)
        o1_imag = torch.zeros([B, N // 2 + 1, self.embed_size], device=x.device)

        o2_real = torch.zeros([B, N // 2 + 1, self.embed_size], device=x.device)
        o2_imag = torch.zeros([B, N // 2 + 1, self.embed_size], device=x.device)

        o1_real = F.relu(
            torch.einsum("bid,d->bid", x.real, self.w[0])
            - torch.einsum("bid,d->bid", x.imag, self.w[1])
            + self.rb1
        )

        o1_imag = F.relu(
            torch.einsum("bid,d->bid", x.imag, self.w[0])
            + torch.einsum("bid,d->bid", x.real, self.w[1])
            + self.ib1
        )

        o2_real = (
            torch.einsum("bid,d->bid", o1_real, self.w1[0])
            - torch.einsum("bid,d->bid", o1_imag, self.w1[1])
            + self.rb2
        )

        o2_imag = (
            torch.einsum("bid,d->bid", o1_imag, self.w1[0])
            + torch.einsum("bid,d->bid", o1_real, self.w1[1])
            + self.ib2
        )

        y = torch.stack([o2_real, o2_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        B, L, N = x.shape
        z = x
        z = self.revin_layer(z, "norm")
        x = z

        x = x.permute(0, 2, 1)
        x = self.embedding(x)  # B, N, D
        x = self.layernorm(x)
        x = torch.fft.rfft(x, dim=1, norm="ortho")

        weight = self.texfilter(x)
        x = x * weight
        x = torch.fft.irfft(x, n=N, dim=1, norm="ortho")
        x = self.layernorm1(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.output(x)
        x = x.permute(0, 2, 1)

        z = x
        z = self.revin_layer(z, "denorm")
        x = z

        return x
