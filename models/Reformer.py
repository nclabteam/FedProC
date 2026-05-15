import torch
import torch.nn as nn

from layers.DataEmbedding import DataEmbedding_wo_pos
from layers.SelfAttention_Family import ReformerLayer
from layers.Transformer_EncDec import Encoder, EncoderLayer


class Reformer(nn.Module):
    """Reformer: The Efficient Transformer. ICLR 2020."""

    optional = {
        "d_model": 512,
        "n_heads": 8,
        "e_layers": 2,
        "d_ff": 2048,
        "dropout": 0.05,
        "activation": "gelu",
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--d_model", type=int, default=None)
        parser.add_argument("--n_heads", type=int, default=None)
        parser.add_argument("--e_layers", type=int, default=None)
        parser.add_argument("--d_ff", type=int, default=None)
        parser.add_argument("--dropout", type=float, default=None)
        parser.add_argument("--activation", type=str, default=None)

    def __init__(self, configs, bucket_size=4, n_hashes=4):
        super().__init__()
        self.pred_len = configs.output_len

        self.enc_embedding = DataEmbedding_wo_pos(configs.input_channels, configs.d_model, embed_type="timeF", dropout=configs.dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ReformerLayer(None, configs.d_model, configs.n_heads, bucket_size=bucket_size, n_hashes=n_hashes),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )
        self.projection = nn.Linear(configs.d_model, configs.output_channels)

    def forward(self, x, **kwargs):
        x_mark = kwargs.get("x_mark", None)
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        x_placeholder = torch.zeros_like(x[:, -self.pred_len :, :])
        x_full = torch.cat([x, x_placeholder], dim=1)
        if x_mark is not None:
            x_mark_full = torch.cat([x_mark, torch.zeros(x.shape[0], self.pred_len, x_mark.shape[-1], device=x.device)], dim=1)
        else:
            x_mark_full = None

        enc_out = self.enc_embedding(x_full, x_mark_full)
        enc_out, _ = self.encoder(enc_out)
        dec_out = self.projection(enc_out)

        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, dec_out.shape[1], 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, dec_out.shape[1], 1)
        return dec_out[:, -self.pred_len :, :]
