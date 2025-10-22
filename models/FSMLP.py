import math
from typing import Optional

import numpy as np
import torch
import torch_dct as dct
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F

from layers import PositionalEncoding, RevIN, SimplexLinear, Transpose

optional = {
    "f_model": 1,
    "e_layers": 3,
    "n_heads": 1,
    "d_model": 256,
    "d_ff": 256,
    "fc_dropout": 0.7,
    "dropout": 0.3,
    "head_dropout": 0.0,
    "add": False,
    "individual": False,
    "wo_conv": False,
    "serial_conv": False,
    "patch_len": [16],
    "kernel_list": [3, 7, 9],
    "period": [96],
    "stride": [1],
    "affine": False,
    "subtract_last": False,
    "d_k": 128,
    "d_v": 128,
    "norm": "BatchNorm",
    "act": "gelu",
    "attn_dropout": 0.0,
    "res_attention": True,
    "pre_norm": False,
    "store_attn": False,
    "pe": "zeros",
    "learn_pe": True,
    "m_model": 1,
    "m_layers": 1,
}


def args_update(parser):
    parser.add_argument("--f_model", type=int, default=None)
    parser.add_argument("--e_layers", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--fc_dropout", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--head_dropout", type=float, default=None)
    parser.add_argument("--add", type=bool, default=None)
    parser.add_argument("--individual", type=bool, default=None)
    parser.add_argument("--wo_conv", type=bool, default=None)
    parser.add_argument("--serial_conv", type=bool, default=None)
    parser.add_argument("--patch_len", type=int, nargs="+", default=None)
    parser.add_argument("--kernel_list", type=int, nargs="+", default=None)
    parser.add_argument("--period", type=int, nargs="+", default=None)
    parser.add_argument("--stride", type=int, nargs="+", default=None)
    parser.add_argument("--affine", type=bool, default=None)
    parser.add_argument("--subtract_last", type=bool, default=None)
    parser.add_argument("--d_k", type=int, default=None)
    parser.add_argument("--d_v", type=int, default=None)
    parser.add_argument(
        "--norm", type=str, default=None, choices=["BatchNorm", "LayerNorm"]
    )
    parser.add_argument(
        "--act", type=str, default=None, choices=["gelu", "relu", "selu"]
    )
    parser.add_argument("--attn_dropout", type=float, default=None)
    parser.add_argument("--res_attention", type=bool, default=None)
    parser.add_argument("--pre_norm", type=bool, default=None)
    parser.add_argument("--store_attn", type=bool, default=None)
    parser.add_argument("--pe", type=str, default=None)
    parser.add_argument("--learn_pe", type=bool, default=None)
    parser.add_argument("--m_model", type=int, default=None)
    parser.add_argument("--m_layers", type=int, default=None)


class FSMLP(nn.Module):
    def __init__(
        self,
        configs,
    ):
        super().__init__()
        self.model = FSMLP_Backbone(
            c_in=configs.input_channels,
            context_window=configs.input_len,
            target_window=configs.output_len,
            wo_conv=configs.wo_conv,
            serial_conv=configs.serial_conv,
            add=configs.add,
            patch_len=configs.patch_len,
            kernel_list=configs.kernel_list,
            period=configs.period,
            stride=configs.stride,
            n_layers=configs.e_layers,
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            d_k=configs.d_k,
            d_v=configs.d_v,
            d_ff=configs.d_ff,
            norm=configs.norm,
            attn_dropout=configs.attn_dropout,
            dropout=configs.dropout,
            act=configs.act,
            res_attention=configs.res_attention,
            pre_norm=configs.pre_norm,
            store_attn=configs.store_attn,
            pe=configs.pe,
            learn_pe=configs.learn_pe,
            fc_dropout=configs.fc_dropout,
            head_dropout=configs.head_dropout,
            m_model=configs.m_model,
            affine=configs.affine,
            m_layers=configs.m_layers,
            subtract_last=configs.subtract_last,
            f_model=configs.f_model,
            individual=configs.individual,
        )

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)
        # x: [Batch, Channel, Input length]
        x = self.model(x)
        x = x.permute(0, 2, 1)
        # x: [Batch, Input length, Channel]
        return x


class FSMLP_Backbone(nn.Module):
    def get_para(self):
        weights = self.linear.weight.data.detach().cpu()
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list(
            "custom_cmap", [(0, "blue"), (0.5, "white"), (1, "red")]
        )

        ax = sns.heatmap(weights, cmap=cmap, center=0, linewidth=0)
        plt.savefig("time.pdf", format="pdf")

    def __init__(
        self,
        c_in: int,
        context_window: int,
        target_window: int,
        period,
        patch_len,
        stride,
        kernel_list,
        serial_conv=False,
        wo_conv=False,
        add=False,
        m_model=512,
        n_layers: int = 3,
        d_model=128,
        n_heads=16,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        act: str = "gelu",
        res_attention: bool = False,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        fc_dropout: float = 0.0,
        head_dropout=0,
        individual=False,
        affine=True,
        subtract_last=False,
        f_model=0,
        m_layers=1,
    ):
        super().__init__()
        self.n = 3
        self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        self.period_list = period
        self.period_len = [math.ceil(context_window / i) for i in self.period_list]
        self.kernel_list = [(n, patch_len[i]) for i, n in enumerate(self.period_len)]
        self.stride_list = [
            (n, m // 2 if stride is None else stride[i])
            for i, (n, m) in enumerate(self.kernel_list)
        ]
        self.d_model = d_model
        self.cin = c_in
        self.dim_list = [k[0] * k[1] for k in self.kernel_list]
        self.tokens_list = [
            (self.period_len[i] // s[0])
            * ((math.ceil(self.period_list[i] / k[1]) * k[1] - k[1]) // s[1] + 1)
            for i, (k, s) in enumerate(zip(self.kernel_list, self.stride_list))
        ]
        self.pad_layer = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        (
                            nn.ConstantPad1d((0, p - context_window % p), 0)
                            if context_window % p != 0
                            else nn.Identity()
                        ),
                        (
                            nn.ConstantPad1d((0, k[1] - p % k[1]), 0)
                            if p % k[1] != 0
                            else nn.Identity()
                        ),
                    ]
                )
                for p, (k, s) in zip(
                    self.period_list, zip(self.kernel_list, self.stride_list)
                )
            ]
        )

        self.backbone = nn.Sequential(
            TSTiEncoder(
                patch_num=sum(self.period_len),
                n_layers=n_layers,
                d_model=d_model,
                n_heads=n_heads,
                d_k=d_k,
                d_v=d_v,
                d_ff=d_ff,
                norm=norm,
                attn_dropout=attn_dropout,
                dropout=dropout,
                act=act,
                res_attention=res_attention,
                pre_norm=pre_norm,
                store_attn=store_attn,
                pe=pe,
                learn_pe=learn_pe,
            ),
            nn.Flatten(start_dim=-2),
            nn.Linear(sum(self.period_len) * d_model, context_window),
        )
        self.last = nn.Linear(context_window, target_window)
        self.wo_conv = wo_conv

        self.serial_conv = serial_conv
        self.compensate = (context_window + target_window) / context_window
        if not self.wo_conv:
            self.conv = nn.Sequential(
                *[
                    nn.Sequential(
                        nn.Conv1d(
                            self.n + 1,
                            self.n + 1,
                            kernel_size=i,
                            groups=self.n + 1,
                            padding="same",
                        ),
                        nn.SELU(),
                        nn.Dropout(fc_dropout),
                        nn.BatchNorm1d(self.n + 1),
                    )
                    for i in kernel_list
                ],
                nn.Flatten(start_dim=-2),
                nn.Linear(context_window * (self.n + 1), context_window),
            )

            self.conv1 = nn.ModuleList(
                [
                    nn.Sequential(
                        *[
                            nn.Sequential(
                                nn.Conv1d(
                                    n, n, kernel_size=i, groups=n, padding="same"
                                ),
                                nn.SELU(),
                                nn.BatchNorm1d(n),
                            )
                            for i in kernel_list
                        ],
                        nn.Dropout(fc_dropout),
                    )
                    for n in self.period_len
                ]
            )
        self.conv_drop = nn.Dropout(fc_dropout)
        self.glo = nn.ModuleList(
            [nn.Linear(context_window, context_window) for i in range(len(period))]
        )
        self.dropout = nn.Dropout(dropout)
        self.mix = channel_mix(c_in, d_model, m_model, f_model, m_layers, fc_dropout)
        self.individual = individual
        if individual == False:
            self.W_P = nn.ModuleList(
                [
                    nn.Linear(self.period_list[i], d_model)
                    for i in range(len(self.period_list))
                ]
            )
            self.W_P1 = nn.ModuleList(
                [
                    nn.Linear(self.period_list[i], d_model)
                    for i in range(len(self.period_list))
                ]
            )

        else:
            self.W_P1 = nn.ModuleList(
                [
                    nn.Linear(self.period_list[i], d_model)
                    for i in range(len(self.period_list))
                ]
            )
            self.loc_W_p1 = nn.ModuleList(
                [
                    nn.ModuleList(
                        [nn.Linear(self.period_list[i], d_model) for _ in range(c_in)]
                    )
                    for i in range(len(self.period_list))
                ]
            )

            self.W_P = nn.ModuleList(
                [
                    nn.Linear(self.period_list[i], d_model)
                    for i in range(len(self.period_list))
                ]
            )
            self.loc_W_p = nn.ModuleList(
                [
                    nn.ModuleList(
                        [nn.Linear(self.period_list[i], d_model) for _ in range(c_in)]
                    )
                    for i in range(len(self.period_list))
                ]
            )

        self.head = Head(
            context_window, 1, target_window, head_dropout=head_dropout, concat=not add
        )
        self.threshold_param = nn.Parameter(torch.rand(1) * 0.5)
        self.linears = nn.ModuleList(
            [nn.Linear(context_window // ((n + 1) * 2), d_model // 2) for n in range(2)]
        )
        self.linear_all = [nn.Linear(d_model * 3, d_model // 4)]

    def decouple(self, z, linear_all, linears, n):
        store = []

        def sub_decouple(z, linears, n, store):
            if n == 0:
                return
            n = n - 1
            index_tensor = torch.arange(z.size(-1))
            odd_indices = index_tensor % 2 != 0
            z_odd = z[:, :, odd_indices]
            z_even = z[:, :, ~odd_indices]

            sub_decouple(z_odd, linears, n, store)
            sub_decouple(z_even, linears, n, store)

            z1 = torch.cat(
                [self.linears[n](dct.dct(z_odd)), self.linears[n](dct.dct(z_even))],
                dim=-1,
            )

            store.append(z1)
            if n == 0:
                return

        sub_decouple(z, linears, n, store)
        res = torch.cat(store, dim=-1)
        return res

    def forward(self, z):
        # z: [bs x nvars x seq_len]
        z = z.permute(0, 2, 1)
        z = self.revin_layer(z, "norm")
        z = z.permute(0, 2, 1)
        res = []

        time_z = z
        z = dct.dct(z)
        for i, period in enumerate(self.period_list):

            time_z = (
                self.pad_layer[i][0](time_z)
                .reshape(z.shape[0], -1, z.shape[1], period)
                .permute(0, 1, 3, 2)
            )

            x = self.pad_layer[i][0](z).reshape(z.shape[0] * z.shape[1], -1, period)

            glo = x  # +loc*F.sigmoid(x)#+loc*F.sigmoid(x)
            glo = rearrange(glo, "(b m) d n -> b m d n", b=z.shape[0]).contiguous()
            if not self.individual:
                glo = self.W_P[i](glo)  # x: [bs x nvars x patch_num x d_model]
            else:
                tmp = []
                tmp = (
                    torch.zeros(
                        (glo.shape[0], glo.shape[1], glo.shape[2], self.d_model)
                    )
                    .to(glo.dtype)
                    .to(glo.device)
                )
                for j in range(self.cin):

                    tmp[:, i, :, :] = self.loc_W_p[i][j](glo[:, i, :, :])
                glo = self.W_P[i](glo) + tmp

            glo = glo.permute(0, 2, 3, 1)
            glo = glo + (self.mix(glo))
            glo = glo.permute(0, 3, 2, 1)

            res.append(glo)
        glo = torch.cat(res, dim=-1)
        glo = self.backbone(glo)
        z = self.last(glo)
        z = dct.idct(z)

        z = z.permute(0, 2, 1)
        z = self.revin_layer(z, "denorm")
        z = z.permute(0, 2, 1)
        return z


class TSTiEncoder(nn.Module):
    def __init__(
        self,
        patch_num,
        n_layers=3,
        d_model=128,
        n_heads=16,
        d_k=None,
        d_v=None,
        d_ff=256,
        norm="BatchNorm",
        attn_dropout=0.0,
        dropout=0.0,
        act="gelu",
        store_attn=False,
        res_attention=False,
        pre_norm=False,
        pe="zeros",
        learn_pe=True,
    ):
        super().__init__()
        res_attention = False

        # Positional encoding
        self.W_pos = PositionalEncoding(pe, learn_pe, patch_num, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(0.1)

        # Encoder
        self.d_model = d_model
        self.encoder = TSTEncoder(
            patch_num,
            d_model,
            n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            pre_norm=pre_norm,
            activation=act,
            res_attention=res_attention,
            n_layers=n_layers,
            store_attn=store_attn,
            pos=self.W_pos,
        )

    def forward(self, x) -> Tensor:
        # x: [bs x nvars x patch_len x patch_num]

        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0, 1, 3, 2)
        # x: [bs x nvars x patch_num x patch_len]

        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # u: [bs * nvars x patch_num x d_model]
        u = self.dropout(u + self.W_pos)
        # u: [bs * nvars x patch_num x d_model]

        # Encoder
        z = u
        z = self.encoder(u)
        # z: [bs * nvars x patch_num x d_model]
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))
        # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0, 1, 3, 2)
        # z: [bs x nvars x d_model x patch_num]

        return z


class TSTEncoder(nn.Module):
    def __init__(
        self,
        q_len,
        d_model,
        n_heads,
        d_k=None,
        d_v=None,
        d_ff=None,
        norm="BatchNorm",
        attn_dropout=0.0,
        dropout=0.0,
        activation="gelu",
        res_attention=False,
        n_layers=1,
        pre_norm=False,
        store_attn=False,
        pos=None,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TSTEncoderLayer(
                    d_model,
                    n_heads=n_heads,
                    d_k=d_k,
                    d_v=d_v,
                    d_ff=d_ff,
                    norm=norm,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    res_attention=res_attention,
                    pre_norm=pre_norm,
                    store_attn=store_attn,
                    pos=pos,
                )
                for i in range(n_layers)
            ]
        )
        self.res_attention = res_attention

    def forward(
        self,
        src: Tensor,
    ):
        output = src
        for mod in self.layers:
            output = mod(output)
        return output


class TSTEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        d_k=None,
        d_v=None,
        d_ff=256,
        store_attn=False,
        norm="BatchNorm",
        attn_dropout=0,
        dropout=0.0,
        bias=True,
        res_attention=False,
        pre_norm=False,
        pos=None,
    ):
        super().__init__()
        assert (
            not d_model % n_heads
        ), f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        self.attn = _MultiheadAttention(
            d_model,
            n_heads,
            d_k,
            d_v,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            res_attention=res_attention,
            pos=pos,
        )
        # Multi-Head attention
        self.res_attention = res_attention

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
            )
            self.norm_attn2 = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
            )
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=bias),
        )

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
            )
            self.norm_ffn2 = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
            )
        else:
            self.norm_ffn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dw_conv = nn.Conv1d(
            d_model, d_model, kernel_size=1, stride=1, padding="same", groups=d_model
        )
        self.conv1 = nn.Linear(d_model, d_model)
        self.conv2 = nn.Linear(d_model, d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn
        self.activation = nn.SELU()

    def forward(
        self,
        src: Tensor,
    ) -> Tensor:
        src2 = self.ff(src)
        # Add & Norm
        # Add: residual connection with residual dropout
        src2 = src + self.dropout_ffn(src2)
        src = src2

        return src


class _MultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        d_k=None,
        d_v=None,
        res_attention=False,
        attn_dropout=0.0,
        proj_dropout=0.0,
        qkv_bias=True,
        lsa=False,
        pos=None,
    ):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        self.pos = pos
        self.P_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.P_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(
            d_model,
            n_heads,
            attn_dropout=attn_dropout,
            res_attention=self.res_attention,
            lsa=lsa,
        )

        # Poject output
        self.to_out = nn.Sequential(
            nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout)
        )

    def forward(
        self,
        Q: Tensor,
        K: Optional[Tensor] = None,
        V: Optional[Tensor] = None,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):

        bs = Q.size(0)
        if K is None:
            K = Q
        if V is None:
            V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)
        # v_s    : [bs x n_heads x q_len x d_v]

        q_p = self.P_Q(self.pos).view(1, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_p = self.P_K(self.pos).view(1, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(
                q_s,
                k_s,
                v_s,
                prev=prev,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                q_p=q_p,
                k_p=k_p,
            )
        else:
            output, attn_weights = self.sdp_attn(
                q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = (
            output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        )
        # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(
        self, d_model, n_heads, attn_dropout=0.0, res_attention=False, lsa=False
    ):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(
            torch.tensor((head_dim * 1) ** -0.5), requires_grad=lsa
        )
        self.lsa = lsa

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        q_p=None,
        k_p=None,
    ):
        """

        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        """

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale
        # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None:
            attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:
            # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:
            # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf
            )

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)
        # output: [bs x n_heads x max_q_len x d_v]

        return output, attn_weights


class Head(nn.Module):
    def __init__(
        self, context_window, num_period, target_window, head_dropout=0, concat=True
    ):
        super().__init__()
        self.concat = concat
        self.linear = nn.Linear(
            context_window * (num_period if concat else 1), target_window
        )
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        # x: [bs x nvars x d_model x patch_num]
        if self.concat:
            x = torch.cat(x, dim=-1)
            x = self.linear(x)
        else:
            x = torch.stack(x, dim=-1)
            x = torch.mean(x, dim=-1)
            x = self.linear(x)
        x = self.dropout(x)
        return x


class channel_mix(nn.Module):
    def __init__(self, c_in, d_model, m_model, f_model, e_layers, dropout):
        super().__init__()
        self.f_model = f_model
        if f_model != 0:
            self.emd_time = nn.Linear(d_model, self.f_model)
            self.out_time = nn.Linear(self.f_model, d_model)
        self.e_layers = e_layers
        self.emd = SimplexLinear(c_in, m_model)
        self.cin = c_in
        self.activation = nn.SELU()
        self.m_model = m_model
        self.hypernetwork = HyperNetwork(64, 128, c_in)
        self.embeddings = nn.Parameter(torch.randn(f_model, 64))
        self.trans_layer = nn.ModuleList(
            [nn.Linear(m_model, m_model) for _ in range(f_model)]
        )
        self.out_layers = SimplexLinear(m_model, c_in)
        self.random_emd = nn.Linear(c_in, m_model)
        self.random_up = nn.Linear(m_model, c_in)
        self.cos_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(f_model, 2 * f_model),
                    nn.SELU(),
                    nn.Dropout(dropout),
                    nn.Linear(f_model * 2, f_model),
                )
                for _ in range(e_layers)
            ]
        )
        self.down = nn.ModuleList([nn.Linear(m_model, m_model) for _ in range(f_model)])
        self.ffn = nn.Linear(10, 10)
        for param in self.random_emd.parameters():
            param.requires_grad = False
        self.random_layers = nn.ModuleList(
            [nn.Linear(m_model, m_model) for _ in range(e_layers)]
        )
        for layer in self.random_layers:
            randomly_fix_parameters(layer, 0.995)
        self.dw_conv = nn.ModuleList(
            [
                nn.Conv1d(
                    f_model,
                    f_model,
                    kernel_size=71,
                    stride=1,
                    padding="same",
                    groups=f_model,
                )
                for _ in range(e_layers)
            ]
        )
        with torch.no_grad():
            conv = self.dw_conv[0]
            conv.weight.fill_(1 / 71)  # Set weights to all 1s
            if conv.bias is not None:
                conv.bias.fill_(0.0)
        self.layers = nn.ModuleList(
            [SimplexLinear(m_model, m_model) for _ in range(e_layers)]
        )
        self.time_layers = nn.ModuleList(
            [nn.Linear(f_model, f_model) for _ in range(e_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(10, m_model)
        self.layer_norms1 = nn.ModuleList([nn.LayerNorm(c_in) for _ in range(e_layers)])
        self.layer_norms2 = nn.ModuleList(
            [nn.LayerNorm(f_model) for _ in range(e_layers)]
        )
        self.perm = [torch.randperm(c_in) for _ in range(10)]

        self.row_layers = nn.ModuleList([nn.Linear(30, 30) for _ in range(e_layers)])
        self.col_layers = nn.ModuleList([nn.Linear(30, 30) for _ in range(e_layers)])

        self.prototypes = torch.randn(5, f_model)
        self.mask = (torch.rand(256, 1, f_model, c_in) < 0.05).float()

    def loss(self):
        loss = 0
        for layer in self.layers:
            loss += layer.loss()
        loss += self.emd.loss()
        loss += self.out_layers.loss()
        return loss * 5e-4

    def forward(self, x):
        self.prototypes = self.prototypes.to(x.device)
        if self.f_model != 0:
            embedding = self.emd_time(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
            embedding = self.emd(embedding)
        else:
            embedding = self.emd(x)

        for i in range(len(self.layers)):
            embedding = embedding + self.dropout(
                self.activation(self.layers[i](embedding))
            )
            embedding = embedding.permute(0, 1, 3, 2)
            embedding = embedding + self.dropout(
                self.activation(self.time_layers[i](embedding))
            )
            embedding = embedding.permute(0, 1, 3, 2)
        if self.f_model != 0:
            out = self.out_time(embedding.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
            out = self.out_layers(out)
        return out


class HyperNetwork(nn.Module):
    def __init__(self, d, hidden_dim, output_dim):
        super(HyperNetwork, self).__init__()
        self.fc1 = nn.Linear(d, hidden_dim)
        # generate weight matrix
        self.fc2 = nn.Linear(hidden_dim, output_dim * output_dim)
        # generate bias
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, embedding):
        x = self.fc1(embedding)
        weights = self.fc2(x)
        biases = self.fc3(x)
        output_dim = self.output_dim
        return weights.view(-1, output_dim, output_dim), biases.view(-1, output_dim)


def randomly_fix_parameters(layer, fixed_ratio=0.8):
    # Fix weight parameters
    weight_np = layer.weight.data.cpu().numpy()
    total_weight_params = weight_np.size
    num_fixed_weight_params = int(total_weight_params * fixed_ratio)
    flat_weight_indices = np.random.choice(
        total_weight_params, num_fixed_weight_params, replace=False
    )
    multi_dim_weight_indices = np.unravel_index(flat_weight_indices, weight_np.shape)
    weight_mask = np.ones(weight_np.shape, dtype=bool)
    weight_mask[multi_dim_weight_indices] = False
    layer.register_buffer("weight_mask", torch.tensor(weight_mask, dtype=torch.bool))

    # Set fixed weight parameters to 0 and prevent their gradient updates
    with torch.no_grad():
        layer.weight[~layer.weight_mask] = 0
    layer.weight.requires_grad = True

    def weight_hook(grad):
        grad[~layer.weight_mask] = 0
        return grad

    layer.weight.register_hook(weight_hook)

    # Fix bias parameters (if bias exists)
    if layer.bias is not None:
        bias_np = layer.bias.data.cpu().numpy()
        total_bias_params = bias_np.size
        num_fixed_bias_params = int(total_bias_params * fixed_ratio)
        flat_bias_indices = np.random.choice(
            total_bias_params, num_fixed_bias_params, replace=False
        )
        bias_mask = np.ones(bias_np.shape, dtype=bool)
        bias_mask[flat_bias_indices] = False
        layer.register_buffer("bias_mask", torch.tensor(bias_mask, dtype=torch.bool))

        # Set fixed bias parameters to 0 and prevent their gradient updates
        with torch.no_grad():
            layer.bias[~layer.bias_mask] = 0
        layer.bias.requires_grad = True

        def bias_hook(grad):
            grad[~layer.bias_mask] = 0
            return grad

        layer.bias.register_hook(bias_hook)
