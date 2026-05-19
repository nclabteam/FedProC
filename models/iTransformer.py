import torch.nn as nn

from layers.DataEmbedding import DataEmbedding_inverted
from layers.RevIN import RevIN
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer


class iTransformer(nn.Module):
    """iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. ICLR 2024."""

    optional = {
        "d_model": 512,
        "n_heads": 8,
        "e_layers": 3,
        "d_ff": 512,
        "factor": 1,
        "dropout": 0.1,
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
        self.pred_len = configs.output_len
        self.revin_layer = RevIN(
            configs.input_channels, affine=False, stdev_detach=False
        )
        self.enc_embedding = DataEmbedding_inverted(
            configs.input_len, configs.d_model, dropout=configs.dropout
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
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
        self.projection = nn.Linear(configs.d_model, configs.output_len, bias=True)

    def forward(self, x, **kwargs):
        # x: [B, input_len, N]
        x_mark = kwargs.get("x_mark", None)

        x = self.revin_layer(x, "norm")

        N = x.shape[2]
        enc_out = self.enc_embedding(x, x_mark)  # [B, N, d_model]
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[
            :, :, :N
        ]  # [B, pred_len, N]

        return self.revin_layer(dec_out, "denorm")
