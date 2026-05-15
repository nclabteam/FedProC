import torch
import torch.nn as nn

from layers.DataEmbedding import DataEmbedding_wo_pos
from layers.ETSformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, Transform


class ETSformer(nn.Module):
    """ETSformer: Exponential Smoothing Transformers for Time-series Forecasting. ICLR 2022."""

    optional = {
        "d_model": 512,
        "n_heads": 8,
        "e_layers": 2,
        "d_layers": 2,
        "d_ff": 2048,
        "top_k": 5,
        "dropout": 0.1,
        "activation": "sigmoid",
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--d_model", type=int, default=None)
        parser.add_argument("--n_heads", type=int, default=None)
        parser.add_argument("--e_layers", type=int, default=None)
        parser.add_argument("--d_layers", type=int, default=None)
        parser.add_argument("--d_ff", type=int, default=None)
        parser.add_argument("--top_k", type=int, default=None)
        parser.add_argument("--dropout", type=float, default=None)
        parser.add_argument("--activation", type=str, default=None)

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.input_len
        self.pred_len = configs.output_len
        enc_in = configs.input_channels
        c_out = configs.output_channels

        assert configs.e_layers == configs.d_layers, "ETSformer requires e_layers == d_layers"

        self.enc_embedding = DataEmbedding_wo_pos(enc_in, configs.d_model, embed_type="timeF", dropout=configs.dropout)
        self.level_proj = nn.Linear(enc_in, c_out) if enc_in != c_out else nn.Identity()

        self.encoder = Encoder(
            [
                EncoderLayer(
                    configs.d_model,
                    configs.n_heads,
                    c_out,
                    configs.input_len,
                    configs.output_len,
                    configs.top_k,
                    dim_feedforward=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ]
        )
        self.decoder = Decoder(
            [
                DecoderLayer(configs.d_model, configs.n_heads, c_out, configs.output_len, dropout=configs.dropout)
                for _ in range(configs.d_layers)
            ]
        )
        self.transform = Transform(sigma=0.2)

    def forward(self, x, **kwargs):
        x_mark = kwargs.get("x_mark", None)
        if self.training:
            with torch.no_grad():
                x = self.transform.transform(x)

        res = self.enc_embedding(x, x_mark)
        level = self.level_proj(x)
        level, growths, seasons = self.encoder(res, level, attn_mask=None)
        growth, season = self.decoder(growths, seasons)
        preds = level[:, -1:] + growth + season
        return preds[:, -self.pred_len :, :]
