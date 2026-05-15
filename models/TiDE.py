import torch
import torch.nn as nn
import torch.nn.functional as F


class _LayerNorm(nn.Module):
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class _ResBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.fc3 = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.ln = _LayerNorm(output_dim)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.dropout(self.fc2(out))
        return self.ln(out + self.fc3(x))


class TiDE(nn.Module):
    """TiDE: Long-term Time Series Forecasting with TiDE. arXiv 2023."""

    optional = {
        "d_model": 256,
        "d_ff": 256,
        "e_layers": 2,
        "d_layers": 2,
        "dropout": 0.3,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--d_model", type=int, default=None)
        parser.add_argument("--d_ff", type=int, default=None)
        parser.add_argument("--e_layers", type=int, default=None)
        parser.add_argument("--d_layers", type=int, default=None)
        parser.add_argument("--dropout", type=float, default=None)

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.input_len
        self.pred_len = configs.output_len
        c_out = configs.output_channels
        hidden = configs.d_model
        dropout = configs.dropout

        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        self.feature_dim = freq_map.get(getattr(configs, "freq", "h"), 4)
        feature_encode_dim = 2

        flatten_dim = self.seq_len + (self.seq_len + self.pred_len) * feature_encode_dim
        self.feature_encoder = _ResBlock(
            self.feature_dim, hidden, feature_encode_dim, dropout
        )
        self.encoders = nn.Sequential(
            _ResBlock(flatten_dim, hidden, hidden, dropout),
            *[
                _ResBlock(hidden, hidden, hidden, dropout)
                for _ in range(configs.e_layers - 1)
            ],
        )
        self.decoders = nn.Sequential(
            *[
                _ResBlock(hidden, hidden, hidden, dropout)
                for _ in range(configs.d_layers - 1)
            ],
            _ResBlock(hidden, hidden, c_out * self.pred_len, dropout),
        )
        self.temporal_decoder = _ResBlock(
            c_out + feature_encode_dim, configs.d_ff, 1, dropout
        )
        self.residual_proj = nn.Linear(self.seq_len, self.pred_len)
        self._feature_encode_dim = feature_encode_dim
        self._c_out = c_out

    def _forecast_channel(self, x_ch, full_mark):
        # x_ch: [B, seq_len], full_mark: [B, seq_len+pred_len, feature_dim]
        feature = self.feature_encoder(full_mark)  # [B, T, fenc_dim]
        enc_in = torch.cat([x_ch, feature.reshape(x_ch.shape[0], -1)], dim=-1)
        hidden = self.encoders(enc_in)
        decoded = self.decoders(hidden).reshape(
            x_ch.shape[0], self.pred_len, self._c_out
        )
        future_feat = feature[:, self.seq_len :, :]  # [B, pred_len, fenc_dim]
        out = self.temporal_decoder(torch.cat([future_feat, decoded], dim=-1)).squeeze(
            -1
        )
        out = out + self.residual_proj(x_ch)
        return out  # [B, pred_len]

    def forward(self, x, **kwargs):
        x_mark = kwargs.get("x_mark", None)
        B, T, N = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        if x_mark is None:
            full_mark = torch.zeros(
                B, T + self.pred_len, self.feature_dim, device=x.device
            )
        else:
            future_zeros = torch.zeros(
                B, self.pred_len, x_mark.shape[-1], device=x.device
            )
            full_mark = torch.cat([x_mark, future_zeros], dim=1)

        out = torch.stack(
            [self._forecast_channel(x[:, :, i], full_mark) for i in range(N)], dim=-1
        )  # [B, pred_len, N]

        out = out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        out = out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return out
