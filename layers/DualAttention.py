import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .AutoAttention import AutoAttention
from .MultiHeadAttention import MultiHeadAttention
from .PositionalEncoding import DataEmbedding


class DualAttention(nn.Module):
    """
    Source:https://github.com/Levi-Ackman/Leddam/blob/main/layers/Leddam.py
    """

    def __init__(
        self, enc_in, seq_len, d_model, dropout, pe_type, kernel_size, n_layers=3
    ):
        super().__init__()
        self.n_layers = n_layers
        self.LD = LD(kernel_size=kernel_size)
        self.channel_attn_blocks = nn.ModuleList(
            [channel_attn_block(enc_in, d_model, dropout) for _ in range(self.n_layers)]
        )
        self.auto_attn_blocks = nn.ModuleList(
            [auto_attn_block(enc_in, d_model, dropout) for _ in range(self.n_layers)]
        )
        self.position_embedder = DataEmbedding(
            pe_type=pe_type, seq_len=seq_len, d_model=d_model, c_in=enc_in
        )

    def forward(self, inp):
        inp = self.position_embedder(inp.permute(0, 2, 1)).permute(0, 2, 1)
        main = self.LD(inp)
        residual = inp - main

        res_1 = residual
        res_2 = residual
        for i in range(self.n_layers):
            res_1 = self.auto_attn_blocks[i](res_1)
        for i in range(self.n_layers):
            res_2 = self.channel_attn_blocks[i](res_2)
        res = res_1 + res_2

        return res, main


class channel_attn_block(nn.Module):
    def __init__(self, enc_in, d_model, dropout):
        super(channel_attn_block, self).__init__()
        self.channel_att_norm = nn.BatchNorm1d(enc_in)
        self.fft_norm = nn.LayerNorm(d_model)
        self.channel_attn = MultiHeadAttention(
            d_model=d_model, n_heads=1, proj_dropout=dropout
        )
        self.fft_layer = nn.Sequential(
            nn.Linear(d_model, int(d_model * 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * 2), d_model),
        )

    def forward(self, residual):
        res_2 = self.channel_att_norm(
            self.channel_attn(residual.permute(0, 2, 1)) + residual.permute(0, 2, 1)
        )
        res_2 = self.fft_norm(self.fft_layer(res_2) + res_2)
        return res_2.permute(0, 2, 1)


class auto_attn_block(nn.Module):
    def __init__(self, enc_in, d_model, dropout):
        super(auto_attn_block, self).__init__()
        self.auto_attn_norm = nn.BatchNorm1d(enc_in)
        self.fft_norm = nn.LayerNorm(d_model)
        self.auto_attn = AutoAttention(P=64, d_model=d_model, proj_dropout=dropout)
        self.fft_layer = nn.Sequential(
            nn.Linear(d_model, int(d_model * 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * 2), d_model),
        )

    def forward(self, residual):
        res_1 = self.auto_attn_norm(
            (self.auto_attn(residual) + residual).permute(0, 2, 1)
        )
        res_1 = self.fft_norm(self.fft_layer(res_1) + res_1)
        return res_1.permute(0, 2, 1)


class LD(nn.Module):
    def __init__(self, kernel_size=25):
        super(LD, self).__init__()
        # Define a shared convolution layers for all channels
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            stride=1,
            padding=int(kernel_size // 2),
            padding_mode="replicate",
            bias=True,
        )
        # Define the parameters for Gaussian initialization
        kernel_size_half = kernel_size // 2
        sigma = 1.0  # 1 for variance
        weights = torch.zeros(1, 1, kernel_size)
        for i in range(kernel_size):
            weights[0, 0, i] = math.exp(-(((i - kernel_size_half) / (2 * sigma)) ** 2))

        # Set the weights of the convolution layer
        self.conv.weight.data = F.softmax(weights, dim=-1)
        self.conv.bias.data.fill_(0.0)

    def forward(self, inp):
        # Permute the input tensor to match the expected shape for 1D convolution (B, N, T)
        inp = inp.permute(0, 2, 1)
        # Split the input tensor into separate channels
        input_channels = torch.split(inp, 1, dim=1)

        # Apply convolution to each channel
        conv_outputs = [self.conv(input_channel) for input_channel in input_channels]

        # Concatenate the channel outputs
        out = torch.cat(conv_outputs, dim=1)
        out = out.permute(0, 2, 1)
        return out
