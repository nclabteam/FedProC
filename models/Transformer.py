import torch
import torch.nn as nn

from layers.DataEmbedding import (
    DataEmbedding,
    DataEmbedding_wo_pos,
    DataEmbedding_wo_pos_temp,
    DataEmbedding_wo_temp,
)
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer


class Transformer(nn.Module):

    optional = {
        "d_model": 512,
        "n_heads": 8,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 2048,
        "factor": 1,
        "dropout": 0.05,
        "activation": "gelu",
        "label_len": 48,
        "embed_type": 0,
        "embed": "timeF",
        "freq": "h",
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--d_model", type=int, default=None)
        parser.add_argument("--n_heads", type=int, default=None)
        parser.add_argument("--e_layers", type=int, default=None)
        parser.add_argument("--d_layers", type=int, default=None)
        parser.add_argument("--d_ff", type=int, default=None)
        parser.add_argument("--factor", type=int, default=None)
        parser.add_argument("--dropout", type=float, default=None)
        parser.add_argument(
            "--activation", type=str, default=None, choices=["gelu", "relu"]
        )
        parser.add_argument("--label_len", type=int, default=None)
        parser.add_argument(
            "--embed_type", type=int, default=None, choices=[0, 1, 2, 3, 4]
        )
        parser.add_argument(
            "--embed", type=str, default=None, choices=["timeF", "fixed", "learned"]
        )
        parser.add_argument(
            "--freq",
            type=str,
            default=None,
            choices=["h", "t", "s", "d", "b", "w", "m"],
        )

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.input_len
        self.pred_len = configs.output_len
        self.label_len = configs.label_len
        enc_in = dec_in = c_out = configs.input_channels

        embed_type = configs.embed_type
        embed = configs.embed
        freq = configs.freq

        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        self.mark_dim = freq_map.get(freq, 4)
        d_model = configs.d_model
        n_heads = configs.n_heads
        e_layers = configs.e_layers
        d_layers = configs.d_layers
        d_ff = configs.d_ff
        factor = configs.factor
        dropout = configs.dropout
        activation = configs.activation

        # Embedding
        embed_cls = {
            0: DataEmbedding,
            1: DataEmbedding,
            2: DataEmbedding_wo_pos,
            3: DataEmbedding_wo_temp,
            4: DataEmbedding_wo_pos_temp,
        }[embed_type]
        self.enc_embedding = embed_cls(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = embed_cls(dec_in, d_model, embed, freq, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True),
        )

    def forward(self, x, x_mark=None, y_mark=None, **kwargs):
        """
        Args:
            x: [B, input_len, channels]
            x_mark: [B, input_len, mark_dim] encoder time features (or None)
            y_mark: [B, pred_len, mark_dim] decoder time features (or None)
        """
        B = x.size(0)
        if x_mark is None:
            x_mark = torch.zeros(B, self.seq_len, self.mark_dim, device=x.device)
        if y_mark is None:
            y_mark = torch.zeros(B, self.pred_len, self.mark_dim, device=x.device)

        # Encoder
        enc_out = self.enc_embedding(x, x_mark)
        enc_out, _ = self.encoder(enc_out)

        # Decoder input: [last label_len of x | zeros for pred_len]
        dec_inp = torch.zeros(B, self.pred_len, x.size(-1), device=x.device)
        dec_inp = torch.cat([x[:, -self.label_len :, :], dec_inp], dim=1)

        # Decoder marks: [last label_len of x_mark | y_mark (pred_len)]
        label_mark = x_mark[:, -self.label_len :, :]
        dec_mark = torch.cat([label_mark, y_mark], dim=1)

        # Decoder
        dec_out = self.dec_embedding(dec_inp, dec_mark)
        dec_out = self.decoder(dec_out, enc_out)

        return dec_out[:, -self.pred_len :, :]
