from math import sqrt

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from layers import PositionalEmbedding

optional = {
    "patch_len": 24,
    "d_model": 512,
    "d_ff": 2048,
    "e_layers": 2,
    "n_heads": 8,
    "dropout": 0.1,
    "factor": 1,
    "activation": "gelu",
}


def args_update(parser):
    parser.add_argument(
        "--patch_len",
        type=int,
        default=None,
        help="input sequence length",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=None,
        help="dimension of model",
    )
    parser.add_argument(
        "--d_ff",
        type=int,
        default=None,
        help="dimension of fcn",
    )
    parser.add_argument(
        "--e_layers", type=int, default=None, help="num of encoder layers"
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=None,
        help="num of attention heads",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="dropout rate",
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=None,
        help="attention factor",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default=None,
        choices=["gelu", "relu"],
        help="activation function",
    )


class Timer(nn.Module):
    """
    Paper: https://arxiv.org/abs/2402.02368
    Source: https://github.com/thuml/Large-Time-Series-Model/blob/main/models/Timer.py
    """

    def __init__(self, configs):
        super().__init__()
        num_patches = (configs.input_len + configs.patch_len - 1) // configs.patch_len

        self.decoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=True,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self.enc_embedding = PatchEmbedding(
            d_model=configs.d_model,
            patch_len=configs.patch_len,
            stride=configs.patch_len,
            padding=0,
            dropout=configs.dropout,
        )
        self.proj = nn.Linear(
            in_features=num_patches * configs.d_model,
            out_features=configs.output_len,
            bias=True,
        )

    def forward(self, x):
        # Input: [batch_size, input_len, input_channels]
        B, L, M = x.shape  # B=batch_size, L=input_len, M=input_channels

        # Normalization from Non-stationary Transformer
        means = x.mean(1, keepdim=True).detach()  # [B, 1, M]
        x = x - means
        var = torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5  # [B, 1, M]
        stdev = torch.sqrt(var).detach()  # [B, 1, M]
        x /= stdev  # [B, L, M]

        # do patching and embedding
        x = x.permute(0, 2, 1)  # [B, M, L] - permute to [batch, channels, length]
        dec_in, n_vars = self.enc_embedding(x)
        # dec_in=[B * M, N, D] where N=num_patches, D=d_model

        # Transformer Blocks
        dec_out, attns = self.decoder(dec_in)  # [B * M, N, D]

        # Reshape to [B, M, N*D] to project all patch info per variable
        dec_out = dec_out.reshape(B, M, -1)  # [B, M, N*D]

        # Project to output length: [B, M, N*D] -> [B, M, output_len]
        dec_out = self.proj(dec_out)  # [B, M, output_len]

        # Transpose to get [batch_size, output_len, output_channels]
        dec_out = dec_out.transpose(1, 2)  # [B, output_len, M]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * stdev + means  # [B, output_len, M]

        # Output: [batch_size, output_len, output_channels]
        return dec_out


class PatchEmbedding(nn.Module):
    def __init__(
        self, d_model, patch_len, stride, padding, dropout, position_embedding=True
    ):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.positioned = position_embedding

        # Positional embedding
        if position_embedding:
            self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]  # [B, M, T]
        x = self.padding_patch_layer(x)
        x = x.unfold(
            dimension=-1, size=self.patch_len, step=self.stride
        )  # [B, M, N, L]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        if self.positioned:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(x)
        return self.dropout(x), n_vars


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        return self._mask


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(
                zip(self.attn_layers, self.conv_layers)
            ):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
