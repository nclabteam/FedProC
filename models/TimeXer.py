import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.DataEmbedding import DataEmbedding_inverted
from layers.PositionalEmbedding import PositionalEmbedding
from layers.RevIN import RevIN
from layers.SelfAttention_Family import AttentionLayer, FullAttention


class _EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class _EncoderLayer(nn.Module):
    def __init__(
        self,
        self_attention,
        cross_attention,
        d_model,
        d_ff=None,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x = self.norm1(x)

        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        x_glb_attn = self.dropout(
            self.cross_attention(x_glb, cross, cross, attn_mask=cross_mask)[0]
        )
        x_glb_attn = torch.reshape(
            x_glb_attn, (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])
        ).unsqueeze(1)
        x_glb = self.norm2(x_glb_ori + x_glb_attn)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class _Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross):
        for layer in self.layers:
            x = layer(x, cross)
        if self.norm is not None:
            x = self.norm(x)
        return x


class TimeXer(nn.Module):
    """TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables. NeurIPS 2024."""

    optional = {
        "d_model": 512,
        "n_heads": 8,
        "e_layers": 2,
        "d_ff": 2048,
        "factor": 1,
        "dropout": 0.1,
        "activation": "gelu",
        "use_norm": 1,
        "patch_len": 16,
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
        parser.add_argument("--use_norm", type=int, default=None)
        parser.add_argument("--patch_len", type=int, default=None)

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.output_len
        self.use_norm = bool(configs.use_norm)
        enc_in = configs.input_channels
        self.revin_layer = RevIN(enc_in, affine=False, stdev_detach=False)

        patch_len = configs.patch_len
        patch_num = configs.input_len // patch_len

        self.en_embedding = _EnEmbedding(
            enc_in, configs.d_model, patch_len, configs.dropout
        )
        self.ex_embedding = DataEmbedding_inverted(
            configs.input_len, configs.d_model, dropout=configs.dropout
        )

        self.encoder = _Encoder(
            [
                _EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        head_nf = configs.d_model * (patch_num + 1)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(head_nf, configs.output_len),
            nn.Dropout(configs.dropout),
        )

    def forward(self, x, **kwargs):
        x_mark = kwargs.get("x_mark", None)

        B, T, N = x.shape
        if self.use_norm:
            x = self.revin_layer(x, "norm")

        en_embed, n_vars = self.en_embedding(x.permute(0, 2, 1))
        ex_embed = self.ex_embedding(x, x_mark)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out).permute(0, 2, 1)  # [B, pred_len, N]

        if self.use_norm:
            dec_out = self.revin_layer(dec_out, "denorm")

        return dec_out
