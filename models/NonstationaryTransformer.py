import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.DataEmbedding import DataEmbedding_wo_pos
from layers.RevIN import RevIN


class _DSAttention(nn.Module):
    """De-stationary Attention."""

    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / math.sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = (
            0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)
        )  # B x 1 x 1 x S

        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous(), None


class _AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super().__init__()
        d_keys = d_model // n_heads
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, _ = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )
        out = out.view(B, L, -1)
        return self.out_projection(out)


class _EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff, dropout, activation="gelu"):
        super().__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = self.norm1(x + self.dropout(new_x))
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class _Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
        if self.norm is not None:
            x = self.norm(x)
        return x


class _Projector(nn.Module):
    """MLP projector to learn de-stationary tau and delta factors."""

    def __init__(self, enc_in, seq_len, hidden_dim, hidden_layers, output_dim):
        super().__init__()
        self.series_conv = nn.Conv1d(
            seq_len, 1, kernel_size=3, padding=1, padding_mode="circular", bias=False
        )
        layers = [nn.Linear(2 * enc_in, hidden_dim), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        B = x.shape[0]
        x = self.series_conv(x)  # B x 1 x E
        x = torch.cat([x, stats], dim=1)  # B x 2 x E
        x = x.view(B, -1)
        return self.backbone(x)


class NonstationaryTransformer(nn.Module):
    """Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting. NeurIPS 2022."""

    optional = {
        "d_model": 512,
        "n_heads": 8,
        "e_layers": 2,
        "d_ff": 2048,
        "factor": 1,
        "dropout": 0.05,
        "activation": "gelu",
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--d_model", type=int, default=None)
        parser.add_argument("--n_heads", type=int, default=None)
        parser.add_argument("--e_layers", type=int, default=None)
        parser.add_argument("--d_ff", type=int, default=None)
        parser.add_argument("--factor", type=int, default=None)
        parser.add_argument("--dropout", type=float, default=None)
        parser.add_argument(
            "--activation", type=str, default=None, choices=["gelu", "relu"]
        )

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.input_len
        self.pred_len = configs.output_len
        enc_in = configs.input_channels
        c_out = configs.output_channels
        p_hidden_dim = 128
        p_hidden_layers = 2
        self.revin_layer = RevIN(enc_in, affine=False)

        self.enc_embedding = DataEmbedding_wo_pos(
            enc_in, configs.d_model, embed_type="timeF", dropout=configs.dropout
        )
        self.encoder = _Encoder(
            [
                _EncoderLayer(
                    _AttentionLayer(
                        _DSAttention(
                            False, configs.factor, attention_dropout=configs.dropout
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    configs.dropout,
                    configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )
        self.projection = nn.Linear(configs.d_model, c_out)
        self.tau_learner = _Projector(
            enc_in, self.seq_len, p_hidden_dim, p_hidden_layers, 1
        )
        self.delta_learner = _Projector(
            enc_in, self.seq_len, p_hidden_dim, p_hidden_layers, self.seq_len
        )

    def forward(self, x, **kwargs):
        x_mark = kwargs.get("x_mark", None)
        x_raw = x.clone().detach()

        x = self.revin_layer(x, "norm")
        stdev = self.revin_layer.stdev
        means = self.revin_layer.mean

        tau = self.tau_learner(x_raw, stdev)  # [B, 1]
        tau = torch.clamp(tau, max=80.0).exp()
        delta = self.delta_learner(x_raw, means)  # [B, seq_len]

        enc_out = self.enc_embedding(x, x_mark)
        enc_out = self.encoder(enc_out, tau=tau, delta=delta)
        dec_out = self.projection(enc_out)[:, -self.pred_len :, :]

        return self.revin_layer(dec_out, "denorm")
