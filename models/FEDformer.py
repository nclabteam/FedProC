import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.AutoCorrelation import AutoCorrelationLayer
from layers.Autoformer_EncDec import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    my_Layernorm,
)
from layers.DataEmbedding import (
    DataEmbedding,
    DataEmbedding_wo_pos,
    DataEmbedding_wo_pos_temp,
    DataEmbedding_wo_temp,
)
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from layers.SeriesDecompMA import SeriesDecompMA, SeriesDecompMultiMA


class FEDformer(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain
    and achieved O(N) complexity.
    """

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
        "moving_avg": 25,
        "version": "Fourier",
        "mode_select": "random",
        "modes": 32,
        "L": 1,
        "base": "legendre",
        "cross_activation": "tanh",
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
        parser.add_argument("--moving_avg", type=int, default=None)
        parser.add_argument(
            "--version", type=str, default=None, choices=["Fourier", "Wavelets"]
        )
        parser.add_argument(
            "--mode_select", type=str, default=None, choices=["random", "else"]
        )
        parser.add_argument("--modes", type=int, default=None)
        parser.add_argument("--L", type=int, default=None)
        parser.add_argument(
            "--base", type=str, default=None, choices=["legendre", "chebyshev"]
        )
        parser.add_argument(
            "--cross_activation", type=str, default=None, choices=["tanh", "softmax"]
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
        moving_avg = configs.moving_avg
        version = configs.version
        mode_select = configs.mode_select
        modes = configs.modes
        L = configs.L
        base = configs.base
        cross_activation = configs.cross_activation

        # Decomp
        if isinstance(moving_avg, list):
            self.decomp = SeriesDecompMultiMA(moving_avg)
        else:
            self.decomp = SeriesDecompMA(moving_avg)

        # Embedding — FEDformer discards position embedding by default
        embed_cls = {
            0: DataEmbedding_wo_pos,
            1: DataEmbedding,
            2: DataEmbedding_wo_pos,
            3: DataEmbedding_wo_temp,
            4: DataEmbedding_wo_pos_temp,
        }[embed_type]
        self.enc_embedding = embed_cls(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = embed_cls(dec_in, d_model, embed, freq, dropout)

        # Attention selection
        if version == "Wavelets":
            encoder_self_att = MultiWaveletTransform(
                ich=d_model, L=L, base=base
            )
            decoder_self_att = MultiWaveletTransform(
                ich=d_model, L=L, base=base
            )
            decoder_cross_att = MultiWaveletCross(
                in_channels=d_model,
                out_channels=d_model,
                seq_len_q=self.seq_len // 2 + self.pred_len,
                seq_len_kv=self.seq_len,
                modes=modes,
                ich=d_model,
                base=base,
                activation=cross_activation,
            )
        else:
            encoder_self_att = FourierBlock(
                in_channels=d_model,
                out_channels=d_model,
                seq_len=self.seq_len,
                modes=modes,
                mode_select_method=mode_select,
                n_heads=n_heads,
            )
            decoder_self_att = FourierBlock(
                in_channels=d_model,
                out_channels=d_model,
                seq_len=self.seq_len // 2 + self.pred_len,
                modes=modes,
                mode_select_method=mode_select,
                n_heads=n_heads,
            )
            decoder_cross_att = FourierCrossAttention(
                in_channels=d_model,
                out_channels=d_model,
                seq_len_q=self.seq_len // 2 + self.pred_len,
                seq_len_kv=self.seq_len,
                modes=modes,
                mode_select_method=mode_select,
                n_heads=n_heads,
            )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(encoder_self_att, d_model, n_heads),
                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model),
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(decoder_self_att, d_model, n_heads),
                    AutoCorrelationLayer(decoder_cross_att, d_model, n_heads),
                    d_model,
                    c_out,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True),
        )

    def forward(self, x, x_mark=None, y_mark=None, **kwargs):
        B = x.size(0)
        if x_mark is None:
            x_mark = torch.zeros(B, self.seq_len, self.mark_dim, device=x.device)
        if y_mark is None:
            y_mark = torch.zeros(B, self.pred_len, self.mark_dim, device=x.device)

        # decomp init
        mean = torch.mean(x, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init, trend_init = self.decomp(x)

        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len :, :], mean], dim=1)
        seasonal_init = F.pad(
            seasonal_init[:, -self.label_len :, :], (0, 0, 0, self.pred_len)
        )

        # decoder marks: [label_len of x_mark | y_mark (pred_len)]
        label_mark = x_mark[:, -self.label_len :, :]
        dec_mark = torch.cat([label_mark, y_mark], dim=1)

        # enc
        enc_out = self.enc_embedding(x, x_mark)
        enc_out, _ = self.encoder(enc_out)

        # dec
        dec_out = self.dec_embedding(seasonal_init, dec_mark)
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, trend=trend_init
        )

        # final
        dec_out = trend_part + seasonal_part
        return dec_out[:, -self.pred_len :, :]
