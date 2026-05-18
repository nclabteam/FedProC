import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.DataEmbedding import DataEmbedding_wo_pos
from layers.MSGBlock import Attention_Block, GraphBlock, Predict


def _fft_for_period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class _ScaleGraphBlock(nn.Module):
    def __init__(
        self,
        seq_len,
        top_k,
        d_model,
        d_ff,
        n_heads,
        dropout,
        c_out,
        conv_channel,
        skip_channel,
        gcn_depth,
        propalpha,
        node_dim,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.k = top_k
        self.att0 = Attention_Block(
            d_model, d_ff, n_heads=n_heads, dropout=dropout, activation="gelu"
        )
        self.norm = nn.LayerNorm(d_model)
        self.gelu = nn.GELU()
        self.gconv = nn.ModuleList(
            [
                GraphBlock(
                    c_out,
                    d_model,
                    conv_channel,
                    skip_channel,
                    gcn_depth,
                    dropout,
                    propalpha,
                    seq_len,
                    node_dim,
                )
                for _ in range(top_k)
            ]
        )

    def forward(self, x):
        B, T, N = x.size()
        scale_list, scale_weight = _fft_for_period(x, self.k)
        res = []
        for i in range(self.k):
            scale = scale_list[i]
            out = self.gconv[i](x)
            if self.seq_len % scale != 0:
                length = ((self.seq_len // scale) + 1) * scale
                padding = torch.zeros(B, length - self.seq_len, N, device=x.device)
                out = torch.cat([out, padding], dim=1)
            else:
                length = self.seq_len
            out = out.reshape(B, length // scale, scale, N).reshape(-1, scale, N)
            out = self.norm(self.att0(out))
            out = self.gelu(out)
            out = out.reshape(B, -1, scale, N).reshape(B, -1, N)
            out = out[:, : self.seq_len, :]
            res.append(out)

        res = torch.stack(res, dim=-1)
        scale_weight = F.softmax(scale_weight, dim=1)
        scale_weight = scale_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * scale_weight, -1)
        return res + x


class MSGNet(nn.Module):
    """MSGNet: Multi-Scale Isometric Graph Network for Time Series Forecasting. AAAI 2024."""

    optional = {
        "d_model": 32,
        "n_heads": 4,
        "d_ff": 64,
        "e_layers": 2,
        "dropout": 0.1,
        "top_k": 3,
        "conv_channel": 32,
        "skip_channel": 32,
        "gcn_depth": 2,
        "propalpha": 0.05,
        "node_dim": 40,
        "individual": 0,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--d_model", type=int, default=None)
        parser.add_argument("--n_heads", type=int, default=None)
        parser.add_argument("--d_ff", type=int, default=None)
        parser.add_argument("--e_layers", type=int, default=None)
        parser.add_argument("--dropout", type=float, default=None)
        parser.add_argument("--top_k", type=int, default=None)
        parser.add_argument("--conv_channel", type=int, default=None)
        parser.add_argument("--skip_channel", type=int, default=None)
        parser.add_argument("--gcn_depth", type=int, default=None)
        parser.add_argument("--propalpha", type=float, default=None)
        parser.add_argument("--node_dim", type=int, default=None)
        parser.add_argument("--individual", type=int, default=None)

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.output_len
        c_out = configs.output_channels

        self.enc_embedding = DataEmbedding_wo_pos(
            configs.input_channels,
            configs.d_model,
            embed_type="timeF",
            dropout=configs.dropout,
        )
        self.model = nn.ModuleList(
            [
                _ScaleGraphBlock(
                    configs.input_len,
                    configs.top_k,
                    configs.d_model,
                    configs.d_ff,
                    configs.n_heads,
                    configs.dropout,
                    c_out,
                    configs.conv_channel,
                    configs.skip_channel,
                    configs.gcn_depth,
                    configs.propalpha,
                    configs.node_dim,
                )
                for _ in range(configs.e_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.projection = nn.Linear(configs.d_model, c_out)
        self.seq2pred = Predict(
            bool(configs.individual),
            c_out,
            configs.input_len,
            configs.output_len,
            configs.dropout,
        )

    def forward(self, x, **kwargs):
        x_mark = kwargs.get("x_mark", None)
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        enc_out = self.enc_embedding(x, x_mark)
        for block in self.model:
            enc_out = self.layer_norm(block(enc_out))

        dec_out = self.projection(enc_out)
        dec_out = self.seq2pred(dec_out.transpose(1, 2)).transpose(1, 2)

        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out[:, -self.pred_len :, :]
