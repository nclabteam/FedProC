import torch
import torch.nn as nn

from layers.DataEmbedding import DataEmbedding
from layers.Pyraformer_Layers import (
    Predictor,
    PyraformerDecoder,
    PyraformerEncoder,
    get_subsequent_mask,
)


class Pyraformer(nn.Module):
    """
    Pyraformer: Pyramidal Attention Mechanism for long-range time series forecasting.
    """

    optional = {
        "d_model": 512,
        "n_heads": 8,
        "e_layers": 2,
        "d_ff": 2048,
        "dropout": 0.05,
        "label_len": 48,
        "embed": "timeF",
        "freq": "h",
        "window_size": [2, 2, 2],
        "inner_size": 5,
        "d_bottleneck": 64,
        "CSCM": "Conv_Construct",
        "decoder": "FC",
        "truncate": True,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--d_model", type=int, default=None)
        parser.add_argument("--n_heads", type=int, default=None)
        parser.add_argument("--e_layers", type=int, default=None)
        parser.add_argument("--d_ff", type=int, default=None)
        parser.add_argument("--dropout", type=float, default=None)
        parser.add_argument("--label_len", type=int, default=None)
        parser.add_argument(
            "--embed", type=str, default=None, choices=["timeF", "fixed", "learned"]
        )
        parser.add_argument(
            "--freq",
            type=str,
            default=None,
            choices=["h", "t", "s", "d", "b", "w", "m"],
        )
        parser.add_argument("--window_size", type=int, nargs="+", default=None)
        parser.add_argument("--inner_size", type=int, default=None)
        parser.add_argument("--d_bottleneck", type=int, default=None)
        parser.add_argument(
            "--CSCM",
            type=str,
            default=None,
            choices=[
                "Conv_Construct",
                "Bottleneck_Construct",
                "MaxPooling_Construct",
                "AvgPooling_Construct",
            ],
        )
        parser.add_argument(
            "--decoder", type=str, default=None, choices=["FC", "attention"]
        )
        parser.add_argument("--truncate", type=int, default=None, choices=[0, 1])

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.input_len
        self.pred_len = configs.output_len
        self.label_len = configs.label_len
        self.channels = configs.input_channels

        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        self.mark_dim = freq_map.get(configs.freq, 4)

        d_model = configs.d_model
        n_head = configs.n_heads
        d_ff = configs.d_ff
        dropout = configs.dropout
        e_layers = configs.e_layers
        window_size = configs.window_size
        inner_size = configs.inner_size
        d_bottleneck = configs.d_bottleneck
        cscm = configs.CSCM
        decoder_type = configs.decoder
        truncate = bool(configs.truncate)

        # Build a namespace for sub-components
        from argparse import Namespace

        opt = Namespace(
            d_model=d_model,
            n_head=n_head,
            d_inner_hid=d_ff,
            n_layer=e_layers,
            dropout=dropout,
            input_len=self.seq_len,
            predict_step=self.pred_len,
            window_size=window_size,
            inner_size=inner_size,
            d_bottleneck=d_bottleneck,
            CSCM=cscm,
            decoder=decoder_type,
            truncate=truncate,
            device="cpu",
        )

        self.decoder_type = decoder_type
        self.enc_embedding = DataEmbedding(
            self.channels, d_model, embed_type="timeF", freq=configs.freq, dropout=dropout
        )

        self.encoder = PyraformerEncoder(opt)

        if decoder_type == "attention":
            # Mask dimensions: Q=label_len+pred_len, K=encoder_out+label_len+pred_len
            enc_out_len = self.seq_len  # encoder output length (before truncation)
            dec_len = self.label_len + self.pred_len
            mask = get_subsequent_mask(
                enc_out_len, window_size, dec_len, truncate
            )
            self.register_buffer("dec_mask", mask)
            self.decoder = PyraformerDecoder(opt, mask)
            self.predictor = Predictor(d_model, self.channels)
            self.dec_embedding = DataEmbedding(
                self.channels, d_model, embed_type="timeF", freq=configs.freq, dropout=dropout
            )
        else:
            self.predictor = Predictor(4 * d_model, self.pred_len * self.channels)

    def forward(self, x, x_mark=None, y_mark=None, **kwargs):
        B = x.size(0)
        if x_mark is None:
            x_mark = torch.zeros(B, self.seq_len, self.mark_dim, device=x.device)

        enc_out = self.enc_embedding(x, x_mark)
        enc_output = self.encoder(enc_out)

        if self.decoder_type == "attention":
            # Build decoder input: last label_len of x + zeros for pred_len
            if y_mark is None:
                y_mark = torch.zeros(B, self.pred_len, self.mark_dim, device=x.device)
            label_mark = x_mark[:, -self.label_len :, :]
            dec_mark = torch.cat([label_mark, y_mark], dim=1)

            dec_inp = torch.zeros(B, self.pred_len, x.size(-1), device=x.device)
            dec_inp = torch.cat([x[:, -self.label_len :, :], dec_inp], dim=1)
            dec_input = self.dec_embedding(dec_inp, dec_mark)

            dec_output = self.decoder(dec_input, enc_output)
            pred = self.predictor(dec_output)[:, -self.pred_len :, :]
        else:
            enc_output = enc_output[:, -1, :]
            pred = self.predictor(enc_output).view(B, self.pred_len, -1)

        return pred
