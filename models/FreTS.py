import torch
import torch.nn as nn
import torch.nn.functional as F

optional = {
    "embed_size": 128,
    "hidden_size": 256,
    "sparsity_threshold": 0.01,
    "scale": 0.02,
}


def args_update(parser):
    parser.add_argument("--embed_size", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--sparsity_threshold", type=float, default=None)
    parser.add_argument("--scale", type=float, default=None)


class FreTS(nn.Module):
    """
    Paper: https://arxiv.org/abs/2311.06184
    Source: https://github.com/aikunyi/FilterNet/blob/main/models/FreTS.py
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.input_len
        self.pred_len = configs.output_len
        self.feature_size = configs.input_channels
        self.embed_size = configs.embed_size
        self.hidden_size = configs.hidden_size
        self.sparsity_threshold = configs.sparsity_threshold
        self.scale = configs.scale

        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.r1 = nn.Parameter(
            self.scale * torch.randn(self.embed_size, self.embed_size)
        )
        self.i1 = nn.Parameter(
            self.scale * torch.randn(self.embed_size, self.embed_size)
        )
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.r2 = nn.Parameter(
            self.scale * torch.randn(self.embed_size, self.embed_size)
        )
        self.i2 = nn.Parameter(
            self.scale * torch.randn(self.embed_size, self.embed_size)
        )
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.fc = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len),
        )

    # dimension extension
    def tokenEmb(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        # N*T*1 x 1*D = N*T*D
        y = self.embeddings
        return x * y

    # frequency temporal learner
    def MLP_temporal(self, x, B, N, L):
        # [B, N, T, D]
        x = torch.fft.rfft(x, dim=2, norm="ortho")  # FFT on L dimension
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=self.seq_len, dim=2, norm="ortho")
        return x

    # frequency channel learner
    def MLP_channel(self, x, B, N, L):
        # [B, N, T, D]
        x = x.permute(0, 2, 1, 3)
        # [B, T, N, D]
        x = torch.fft.rfft(x, dim=2, norm="ortho")  # FFT on N dimension
        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm="ortho")
        x = x.permute(0, 2, 1, 3)
        # [B, N, T, D]
        return x

    # frequency-domain MLPs
    # dimension: FFT along the dimension, r: the real part of weights, i: the imaginary part of weights
    # rb: the real part of bias, ib: the imaginary part of bias
    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        o1_real = torch.zeros(
            [B, nd, dimension // 2 + 1, self.embed_size], device=x.device
        )
        o1_imag = torch.zeros(
            [B, nd, dimension // 2 + 1, self.embed_size], device=x.device
        )

        o1_real = F.relu(
            torch.einsum("bijd,dd->bijd", x.real, r)
            - torch.einsum("bijd,dd->bijd", x.imag, i)
            + rb
        )

        o1_imag = F.relu(
            torch.einsum("bijd,dd->bijd", x.imag, r)
            + torch.einsum("bijd,dd->bijd", x.real, i)
            + ib
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def forward(self, x_enc):
        # x: [Batch, Input length, Channel]
        B, T, N = x_enc.shape
        # embedding x: [B, N, T, D]
        x = self.tokenEmb(x_enc)
        bias = x
        # [B, N, T, D]
        # if self.channel_independence == '1':
        #     x = self.MLP_channel(x, B, N, T)
        # [B, N, T, D]
        x = self.MLP_temporal(x, B, N, T)
        x = x + bias
        x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)
        return x[:, -self.pred_len :, :]  # [B, L, D]
