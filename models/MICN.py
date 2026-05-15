import torch
import torch.nn as nn

from layers.DataEmbedding import DataEmbedding_wo_pos
from layers.SeriesDecompMA import SeriesDecompMA, SeriesDecompMultiMA


class _MIC(nn.Module):
    """MIC layer to extract local and global features via multi-scale isometric convolutions."""

    def __init__(
        self,
        feature_size,
        n_heads,
        dropout,
        decomp_kernel,
        conv_kernel,
        isometric_kernel,
    ):
        super().__init__()
        self.conv_kernel = conv_kernel
        self.isometric_conv = nn.ModuleList(
            [
                nn.Conv1d(
                    feature_size, feature_size, kernel_size=i, padding=0, stride=1
                )
                for i in isometric_kernel
            ]
        )
        self.conv = nn.ModuleList(
            [
                nn.Conv1d(
                    feature_size, feature_size, kernel_size=i, padding=i // 2, stride=i
                )
                for i in conv_kernel
            ]
        )
        self.conv_trans = nn.ModuleList(
            [
                nn.ConvTranspose1d(
                    feature_size, feature_size, kernel_size=i, padding=0, stride=i
                )
                for i in conv_kernel
            ]
        )
        self.decomp = nn.ModuleList([SeriesDecompMA(k) for k in decomp_kernel])
        self.merge = nn.Conv2d(
            feature_size, feature_size, kernel_size=(len(conv_kernel), 1)
        )
        self.conv1 = nn.Conv1d(feature_size, feature_size * 4, kernel_size=1)
        self.conv2 = nn.Conv1d(feature_size * 4, feature_size, kernel_size=1)
        self.norm1 = nn.LayerNorm(feature_size)
        self.norm2 = nn.LayerNorm(feature_size)
        self.norm = nn.LayerNorm(feature_size)
        self.act = nn.Tanh()
        self.drop = nn.Dropout(dropout)

    def _conv_trans_conv(self, inp, conv1d, conv1d_trans, isometric):
        seq_len = inp.shape[1]
        x = inp.permute(0, 2, 1)
        x1 = self.drop(self.act(conv1d(x)))
        zeros = torch.zeros(
            x1.shape[0], x1.shape[1], x1.shape[2] - 1, device=inp.device
        )
        x = torch.cat((zeros, x1), dim=-1)
        x = self.drop(self.act(isometric(x)))
        x = self.norm((x + x1).permute(0, 2, 1)).permute(0, 2, 1)
        x = self.drop(self.act(conv1d_trans(x)))
        x = x[:, :, :seq_len]
        return self.norm(x.permute(0, 2, 1) + inp)

    def forward(self, src):
        multi = []
        for i in range(len(self.conv_kernel)):
            src_out, _ = self.decomp[i](src)
            src_out = self._conv_trans_conv(
                src_out, self.conv[i], self.conv_trans[i], self.isometric_conv[i]
            )
            multi.append(src_out)
        mg = torch.stack(multi, dim=1)  # [B, n_scales, T, C]
        mg = self.merge(mg.permute(0, 3, 1, 2)).squeeze(-2).permute(0, 2, 1)
        y = self.norm1(mg)
        y = self.conv2(self.conv1(y.transpose(-1, 1))).transpose(-1, 1)
        return self.norm2(mg + y)


class _SeasonalPrediction(nn.Module):
    def __init__(
        self,
        embedding_size,
        n_heads,
        dropout,
        d_layers,
        decomp_kernel,
        c_out,
        conv_kernel,
        isometric_kernel,
    ):
        super().__init__()
        self.mic = nn.ModuleList(
            [
                _MIC(
                    embedding_size,
                    n_heads,
                    dropout,
                    decomp_kernel,
                    conv_kernel,
                    isometric_kernel,
                )
                for _ in range(d_layers)
            ]
        )
        self.projection = nn.Linear(embedding_size, c_out)

    def forward(self, dec):
        for mic_layer in self.mic:
            dec = mic_layer(dec)
        return self.projection(dec)


class MICN(nn.Module):
    """MICN: Multi-scale Isometric Convolution Network for Long-term Time Series Forecasting. ICLR 2023."""

    optional = {
        "d_model": 512,
        "n_heads": 8,
        "d_layers": 1,
        "dropout": 0.05,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--d_model", type=int, default=None)
        parser.add_argument("--n_heads", type=int, default=None)
        parser.add_argument("--d_layers", type=int, default=None)
        parser.add_argument("--dropout", type=float, default=None)

    def __init__(self, configs, conv_kernel=(12, 16)):
        super().__init__()
        self.seq_len = configs.input_len
        self.pred_len = configs.output_len
        enc_in = configs.input_channels
        c_out = configs.output_channels

        decomp_kernel = []
        isometric_kernel = []
        for ii in conv_kernel:
            if ii % 2 == 0:
                decomp_kernel.append(ii + 1)
                isometric_kernel.append((self.seq_len + self.pred_len + ii) // ii)
            else:
                decomp_kernel.append(ii)
                isometric_kernel.append((self.seq_len + self.pred_len + ii - 1) // ii)

        self.decomp_multi = SeriesDecompMultiMA(decomp_kernel)
        self.dec_embedding = DataEmbedding_wo_pos(
            enc_in, configs.d_model, embed_type="timeF", dropout=configs.dropout
        )
        self.conv_trans = _SeasonalPrediction(
            embedding_size=configs.d_model,
            n_heads=configs.n_heads,
            dropout=configs.dropout,
            d_layers=configs.d_layers,
            decomp_kernel=decomp_kernel,
            c_out=c_out,
            conv_kernel=list(conv_kernel),
            isometric_kernel=isometric_kernel,
        )
        self.regression = nn.Linear(self.seq_len, self.pred_len)
        nn.init.constant_(self.regression.weight, 1.0 / self.pred_len)

    def forward(self, x, **kwargs):
        x_mark = kwargs.get("x_mark", None)
        seasonal_init, trend = self.decomp_multi(x)
        trend = self.regression(trend.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # [B, pred_len, N]

        zeros = torch.zeros(x.shape[0], self.pred_len, x.shape[2], device=x.device)
        seasonal_dec = torch.cat([seasonal_init[:, -self.seq_len :, :], zeros], dim=1)

        if x_mark is not None:
            zeros_mark = torch.zeros(
                x.shape[0], self.pred_len, x_mark.shape[-1], device=x.device
            )
            full_mark = torch.cat([x_mark, zeros_mark], dim=1)
        else:
            full_mark = None

        dec_out = self.dec_embedding(seasonal_dec, full_mark)
        dec_out = self.conv_trans(dec_out)
        dec_out = dec_out[:, -self.pred_len :, :] + trend[:, -self.pred_len :, :]
        return dec_out
