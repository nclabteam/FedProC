import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Conv_Blocks import Inception_Block_V1
from layers.DataEmbedding import DataEmbedding_wo_pos


def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, seq_len, pred_len, top_k, d_model, d_ff, num_kernels):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels),
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            total = self.seq_len + self.pred_len
            if total % period != 0:
                length = ((total // period) + 1) * period
                padding = torch.zeros(
                    B, length - total, N, device=x.device, dtype=x.dtype
                )
                out = torch.cat([x, padding], dim=1)
            else:
                length = total
                out = x
            out = (
                out.reshape(B, length // period, period, N)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :total, :])

        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        return res + x


class TimesNet(nn.Module):
    """TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis. ICLR 2023."""

    optional = {
        "d_model": 64,
        "d_ff": 64,
        "e_layers": 2,
        "top_k": 5,
        "num_kernels": 6,
        "dropout": 0.1,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--d_model", type=int, default=None)
        parser.add_argument("--d_ff", type=int, default=None)
        parser.add_argument("--e_layers", type=int, default=None)
        parser.add_argument("--top_k", type=int, default=None)
        parser.add_argument("--num_kernels", type=int, default=None)
        parser.add_argument("--dropout", type=float, default=None)

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.input_len
        self.pred_len = configs.output_len
        enc_in = configs.input_channels
        c_out = configs.output_channels

        self.model = nn.ModuleList(
            [
                TimesBlock(
                    configs.input_len,
                    configs.output_len,
                    configs.top_k,
                    configs.d_model,
                    configs.d_ff,
                    configs.num_kernels,
                )
                for _ in range(configs.e_layers)
            ]
        )
        self.enc_embedding = DataEmbedding_wo_pos(
            enc_in, configs.d_model, embed_type="timeF", dropout=configs.dropout
        )
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.predict_linear = nn.Linear(
            configs.input_len, configs.output_len + configs.input_len
        )
        self.projection = nn.Linear(configs.d_model, c_out, bias=True)

    def forward(self, x, **kwargs):
        x_mark = kwargs.get("x_mark", None)

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        enc_out = self.enc_embedding(x, x_mark)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)

        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        dec_out = self.projection(enc_out)

        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(
            1, self.pred_len + self.seq_len, 1
        )
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(
            1, self.pred_len + self.seq_len, 1
        )
        return dec_out[:, -self.pred_len :, :]
