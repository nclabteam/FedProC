import math

import torch
import torch.nn as nn
from einops import rearrange

from layers.RevIN import RevIN
from layers.SelfAttention_Family import AttentionLayer, FullAttention


class _FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_model)
        )

    def forward(self, x):
        return self.net(x)


class _EncoderBlock(nn.Module):
    def __init__(self, d_model, mha, d_hidden, dropout=0.0, channel_wise=False):
        super().__init__()
        self.channel_wise = channel_wise
        if channel_wise:
            self.conv = nn.Conv1d(
                d_model, d_model, kernel_size=1, padding_mode="reflect"
            )
        self.mha = mha
        self.ff = _FeedForward(d_model, d_hidden)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        q = residual
        if self.channel_wise:
            x_r = self.conv(x.permute(0, 2, 1)).transpose(1, 2)
            k = v = x_r
        else:
            k = v = residual
        x, _ = self.mha(q, k, v, attn_mask=None)
        x = self.norm1(self.dropout(x) + residual)
        residual = x
        x = self.norm2(self.dropout(self.ff(residual)) + residual)
        return x


class MultiPatchFormer(nn.Module):
    """MultiPatchFormer: Multi-scale Patch Transformer for Time Series Forecasting."""

    optional = {
        "d_model": 512,
        "n_heads": 8,
        "d_ff": 512,
        "e_layers": 2,
        "dropout": 0.0,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--d_model", type=int, default=None)
        parser.add_argument("--n_heads", type=int, default=None)
        parser.add_argument("--d_ff", type=int, default=None)
        parser.add_argument("--e_layers", type=int, default=None)
        parser.add_argument("--dropout", type=float, default=None)

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.output_len
        self.d_channel = configs.input_channels
        self.revin_layer = RevIN(
            configs.input_channels, affine=False, stdev_detach=False
        )
        N = configs.e_layers
        d_model = configs.d_model
        d_hidden = configs.d_ff
        n_heads = configs.n_heads
        dropout = configs.dropout
        seq_len = configs.input_len

        stride1, patch_len1 = 8, 8
        stride2, patch_len2 = 8, 16
        stride3, patch_len3 = 7, 24
        stride4, patch_len4 = 6, 32

        self.patch_num1 = int((seq_len - patch_len2) // stride2) + 2
        self.padding1 = nn.ReplicationPad1d((0, stride1))
        self.padding2 = nn.ReplicationPad1d((0, stride2))
        self.padding3 = nn.ReplicationPad1d((0, stride3))
        self.padding4 = nn.ReplicationPad1d((0, stride4))

        self.embed1 = nn.Conv1d(1, d_model // 4, kernel_size=patch_len1, stride=stride1)
        self.embed2 = nn.Conv1d(1, d_model // 4, kernel_size=patch_len2, stride=stride2)
        self.embed3 = nn.Conv1d(1, d_model // 4, kernel_size=patch_len3, stride=stride3)
        self.embed4 = nn.Conv1d(1, d_model // 4, kernel_size=patch_len4, stride=stride4)

        pe = torch.zeros(self.patch_num1, d_model)
        for pos in range(self.patch_num1):
            for i in range(0, d_model, 2):
                w = 10000 ** ((2 * i) / d_model)
                pe[pos, i] = math.sin(pos / w)
                pe[pos, i + 1] = math.cos(pos / w)
        self.register_buffer("pe", pe.unsqueeze(0))

        shared_mha = nn.ModuleList(
            [
                AttentionLayer(FullAttention(mask_flag=True), d_model, n_heads)
                for _ in range(N)
            ]
        )
        shared_mha_ch = nn.ModuleList(
            [
                AttentionLayer(FullAttention(mask_flag=True), d_model, n_heads)
                for _ in range(N)
            ]
        )

        self.enc_layers = nn.ModuleList(
            [
                _EncoderBlock(
                    d_model, shared_mha[i], d_hidden, dropout, channel_wise=False
                )
                for i in range(N)
            ]
        )
        self.enc_layers_ch = nn.ModuleList(
            [
                _EncoderBlock(
                    d_model, shared_mha_ch[0], d_hidden, dropout, channel_wise=True
                )
                for _ in range(N)
            ]
        )

        self.embed_ch = nn.Conv1d(d_model * self.patch_num1, d_model, kernel_size=1)

        p8 = self.pred_len // 8
        p_last = self.pred_len - 7 * p8
        self.out1 = nn.Linear(d_model, p8)
        self.out2 = nn.Linear(d_model + p8, p8)
        self.out3 = nn.Linear(d_model + 2 * p8, p8)
        self.out4 = nn.Linear(d_model + 3 * p8, p8)
        self.out5 = nn.Linear(d_model + 4 * p8, p8)
        self.out6 = nn.Linear(d_model + 5 * p8, p8)
        self.out7 = nn.Linear(d_model + 6 * p8, p8)
        self.out8 = nn.Linear(d_model + 7 * p8, p_last)

    def forward(self, x, **kwargs):
        x = self.revin_layer(x, "norm")

        x_i = x.permute(0, 2, 1)  # [B, C, T]
        bc_l = rearrange(x_i, "b c l -> (b c) l").unsqueeze(-2)  # [B*C, 1, T]

        e1 = self.embed1(bc_l).permute(0, 2, 1)
        e2 = self.embed2(self.padding2(bc_l)).permute(0, 2, 1)
        e3 = self.embed3(self.padding3(bc_l)).permute(0, 2, 1)
        e4 = self.embed4(self.padding4(bc_l)).permute(0, 2, 1)

        enc = (
            torch.cat([e1, e2, e3, e4], dim=-1) + self.pe
        )  # [B*C, patch_num1, d_model]

        for layer in self.enc_layers:
            enc = layer(enc)

        B = x.shape[0]
        x_patch_c = rearrange(enc, "(b c) p d -> b c (p d)", b=B, c=self.d_channel)
        x_ch = self.embed_ch(x_patch_c.permute(0, 2, 1)).transpose(
            1, 2
        )  # [B, C, d_model]

        h = self.enc_layers_ch[0](x_ch)

        f1 = self.out1(h)
        f2 = self.out2(torch.cat([h, f1], dim=-1))
        f3 = self.out3(torch.cat([h, f1, f2], dim=-1))
        f4 = self.out4(torch.cat([h, f1, f2, f3], dim=-1))
        f5 = self.out5(torch.cat([h, f1, f2, f3, f4], dim=-1))
        f6 = self.out6(torch.cat([h, f1, f2, f3, f4, f5], dim=-1))
        f7 = self.out7(torch.cat([h, f1, f2, f3, f4, f5, f6], dim=-1))
        f8 = self.out8(torch.cat([h, f1, f2, f3, f4, f5, f6, f7], dim=-1))

        out = torch.cat([f1, f2, f3, f4, f5, f6, f7, f8], dim=-1).permute(
            0, 2, 1
        )  # [B, pred_len, C]

        return self.revin_layer(out, "denorm")
