import torch
import torch.nn as nn

from layers.DataEmbedding import DataEmbedding_wo_pos
from layers.RevIN import RevIN as Normalize
from layers.SeriesDecompMA import SeriesDecompMA


class DFT_series_decomp(nn.Module):
    def __init__(self, top_k: int = 5):
        super().__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x, dim=1)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, _ = torch.topk(freq, k=self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf, dim=1)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    def __init__(self, seq_len, down_sampling_layers, down_sampling_window):
        super().__init__()
        self.down_sampling_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        seq_len // (down_sampling_window**i),
                        seq_len // (down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    nn.Linear(
                        seq_len // (down_sampling_window ** (i + 1)),
                        seq_len // (down_sampling_window ** (i + 1)),
                    ),
                )
                for i in range(down_sampling_layers)
            ]
        )

    def forward(self, season_list):
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    def __init__(self, seq_len, down_sampling_layers, down_sampling_window):
        super().__init__()
        self.up_sampling_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        seq_len // (down_sampling_window ** (i + 1)),
                        seq_len // (down_sampling_window**i),
                    ),
                    nn.GELU(),
                    nn.Linear(
                        seq_len // (down_sampling_window**i),
                        seq_len // (down_sampling_window**i),
                    ),
                )
                for i in reversed(range(down_sampling_layers))
            ]
        )

    def forward(self, trend_list):
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        d_model,
        d_ff,
        dropout,
        channel_independence,
        decomp_method,
        moving_avg,
        top_k,
        down_sampling_layers,
        down_sampling_window,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.down_sampling_window = down_sampling_window
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.channel_independence = channel_independence

        if decomp_method == "moving_avg":
            self.decompsition = SeriesDecompMA(moving_avg)
        elif decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(top_k)
        else:
            raise ValueError(f"Unknown decomp_method: {decomp_method}")

        if not channel_independence:
            self.cross_layer = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model),
            )

        self.mixing_multi_scale_season = MultiScaleSeasonMixing(
            seq_len, down_sampling_layers, down_sampling_window
        )
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(
            seq_len, down_sampling_layers, down_sampling_window
        )

        self.out_cross_layer = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x_list):
        length_list = [x.size(1) for x in x_list]

        season_list, trend_list = [], []
        for x in x_list:
            season, trend = self.decompsition(x)
            if not self.channel_independence:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        out_season_list = self.mixing_multi_scale_season(season_list)
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(
            x_list, out_season_list, out_trend_list, length_list
        ):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class TimeMixer(nn.Module):
    """TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting. ICLR 2024."""

    optional = {
        "d_model": 32,
        "d_ff": 32,
        "e_layers": 3,
        "dropout": 0.1,
        "channel_independence": 1,
        "decomp_method": "moving_avg",
        "moving_avg": 25,
        "down_sampling_layers": 3,
        "down_sampling_window": 2,
        "down_sampling_method": "avg",
        "use_norm": 1,
        "top_k": 5,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--d_model", type=int, default=None)
        parser.add_argument("--d_ff", type=int, default=None)
        parser.add_argument("--e_layers", type=int, default=None)
        parser.add_argument("--dropout", type=float, default=None)
        parser.add_argument("--channel_independence", type=int, default=None)
        parser.add_argument("--decomp_method", type=str, default=None)
        parser.add_argument("--moving_avg", type=int, default=None)
        parser.add_argument("--down_sampling_layers", type=int, default=None)
        parser.add_argument("--down_sampling_window", type=int, default=None)
        parser.add_argument("--down_sampling_method", type=str, default=None)
        parser.add_argument("--use_norm", type=int, default=None)
        parser.add_argument("--top_k", type=int, default=None)

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.input_len
        self.pred_len = configs.output_len
        self.down_sampling_window = configs.down_sampling_window
        self.down_sampling_layers = configs.down_sampling_layers
        self.channel_independence = bool(configs.channel_independence)
        enc_in = configs.input_channels
        c_out = configs.output_channels

        self.pdm_blocks = nn.ModuleList(
            [
                PastDecomposableMixing(
                    seq_len=configs.input_len,
                    pred_len=configs.output_len,
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    channel_independence=self.channel_independence,
                    decomp_method=configs.decomp_method,
                    moving_avg=configs.moving_avg,
                    top_k=configs.top_k,
                    down_sampling_layers=configs.down_sampling_layers,
                    down_sampling_window=configs.down_sampling_window,
                )
                for _ in range(configs.e_layers)
            ]
        )

        self.preprocess = SeriesDecompMA(configs.moving_avg)

        if self.channel_independence:
            self.enc_embedding = DataEmbedding_wo_pos(
                1, configs.d_model, embed_type="timeF", dropout=configs.dropout
            )
        else:
            self.enc_embedding = DataEmbedding_wo_pos(
                enc_in, configs.d_model, embed_type="timeF", dropout=configs.dropout
            )

        self.normalize_layers = nn.ModuleList(
            [
                Normalize(
                    enc_in,
                    affine=True,
                    non_norm=(configs.use_norm == 0),
                )
                for _ in range(configs.down_sampling_layers + 1)
            ]
        )

        self.predict_layers = nn.ModuleList(
            [
                nn.Linear(
                    configs.input_len // (configs.down_sampling_window**i),
                    configs.output_len,
                )
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        if self.channel_independence:
            self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)
        else:
            self.projection_layer = nn.Linear(configs.d_model, c_out, bias=True)
            self.out_res_layers = nn.ModuleList(
                [
                    nn.Linear(
                        configs.input_len // (configs.down_sampling_window**i),
                        configs.input_len // (configs.down_sampling_window**i),
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )
            self.regression_layers = nn.ModuleList(
                [
                    nn.Linear(
                        configs.input_len // (configs.down_sampling_window**i),
                        configs.output_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

        self._c_out = c_out
        self.down_sampling_method = configs.down_sampling_method

        if configs.down_sampling_method == "max":
            self._down_pool = nn.MaxPool1d(
                configs.down_sampling_window, return_indices=False
            )
        elif configs.down_sampling_method == "conv":
            self._down_pool = nn.Conv1d(
                enc_in,
                enc_in,
                kernel_size=3,
                padding=1,
                stride=configs.down_sampling_window,
                padding_mode="circular",
                bias=False,
            )
        else:
            self._down_pool = nn.AvgPool1d(configs.down_sampling_window)

    def _out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = self.out_res_layers[i](out_res.permute(0, 2, 1))
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        return dec_out + out_res

    def _pre_enc(self, x_list):
        if self.channel_independence:
            return x_list, None
        out1, out2 = [], []
        for x in x_list:
            x1, x2 = self.preprocess(x)
            out1.append(x1)
            out2.append(x2)
        return out1, out2

    def _multi_scale_inputs(self, x_enc, x_mark_enc):
        if self.down_sampling_layers == 0:
            return [x_enc], ([x_mark_enc] if x_mark_enc is not None else None)

        x_enc_list = [x_enc]
        x_mark_list = [x_mark_enc] if x_mark_enc is not None else None
        x_enc_t = x_enc.permute(0, 2, 1)
        x_mark_cur = x_mark_enc

        for _ in range(self.down_sampling_layers):
            x_enc_t = self._down_pool(x_enc_t)
            x_enc_list.append(x_enc_t.permute(0, 2, 1))
            if x_mark_enc is not None:
                x_mark_cur = x_mark_cur[:, :: self.down_sampling_window, :]
                x_mark_list.append(x_mark_cur)

        return x_enc_list, x_mark_list

    def forward(self, x, **kwargs):
        x_mark = kwargs.get("x_mark", None)
        x_enc_list, x_mark_list = self._multi_scale_inputs(x, x_mark)

        B = x.shape[0]
        x_list_norm = []
        x_mark_out = []

        if x_mark_list is not None:
            for i, (xi, xm) in enumerate(zip(x_enc_list, x_mark_list)):
                _, T, N = xi.size()
                xi = self.normalize_layers[i](xi, "norm")
                if self.channel_independence:
                    xi = xi.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    xm = xm.repeat(N, 1, 1)
                x_list_norm.append(xi)
                x_mark_out.append(xm)
        else:
            for i, xi in enumerate(x_enc_list):
                _, T, N = xi.size()
                xi = self.normalize_layers[i](xi, "norm")
                if self.channel_independence:
                    xi = xi.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list_norm.append(xi)

        enc_out_list = []
        x_pre, x_res = self._pre_enc(x_list_norm)
        if x_mark_list is not None:
            for xi, xm in zip(x_pre, x_mark_out):
                enc_out_list.append(self.enc_embedding(xi, xm))
        else:
            for xi in x_pre:
                enc_out_list.append(self.enc_embedding(xi, None))

        for block in self.pdm_blocks:
            enc_out_list = block(enc_out_list)

        # Future multi-predictor mixing
        dec_out_list = []
        if self.channel_independence:
            for i, enc_out in enumerate(enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1
                )
                dec_out = self.projection_layer(dec_out)
                dec_out = (
                    dec_out.reshape(B, self._c_out, self.pred_len)
                    .permute(0, 2, 1)
                    .contiguous()
                )
                dec_out_list.append(dec_out)
        else:
            for i, (enc_out, out_res) in enumerate(zip(enc_out_list, x_res)):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1
                )
                dec_out = self._out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, "denorm")
        return dec_out
