import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from layers import RevIN, SeriesDecompMA, TSTiEncoder

optional = {
    "e_layers": 3,
    "n_heads": 4,
    "d_model": 16,
    "d_ff": 128,
    "dropout": 0.3,
    "fc_dropout": 0.3,
    "head_dropout": 0.0,
    "patch_len": 16,
    "stride": 8,
    "padding_patch": "end",
    "individual": False,
    "decomposition": False,
    "kernel_size": 25,
    "revin": True,
    "affine": False,
    "subtract_last": False,
    "max_seq_len": 1024,
    "d_k": None,
    "d_v": None,
    "norm": "BatchNorm",
    "attn_dropout": 0.0,
    "act": "gelu",
    "key_padding_mask": "auto",
    "padding_var": None,
    "attn_mask": None,
    "res_attention": True,
    "pre_norm": False,
    "store_attn": False,
    "pe": "zeros",
    "learn_pe": True,
    "pretrain_head": False,
    "head_type": "flatten",
    "verbose": False,
}


def args_update(parser):
    parser.add_argument("--e_layers", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--fc_dropout", type=float, default=None)
    parser.add_argument("--head_dropout", type=float, default=None)
    parser.add_argument("--patch_len", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--padding_patch", type=str, default=None, choices=["end"])
    parser.add_argument("--individual", type=bool, default=None)
    parser.add_argument("--decomposition", type=bool, default=None)
    parser.add_argument("--kernel_size", type=int, default=None)
    parser.add_argument("--revin", type=bool, default=None)
    parser.add_argument("--affine", type=bool, default=None)
    parser.add_argument("--subtract_last", type=bool, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--d_k", type=int, default=None)
    parser.add_argument("--d_v", type=int, default=None)
    parser.add_argument(
        "--norm", type=str, default=None, choices=["BatchNorm", "LayerNorm"]
    )
    parser.add_argument("--attn_dropout", type=float, default=None)
    parser.add_argument("--act", type=str, default=None, choices=["relu", "gelu"])
    parser.add_argument("--key_padding_mask", type=str, default=None)
    parser.add_argument("--padding_var", type=int, default=None)
    parser.add_argument("--attn_mask", type=str, default=None)
    parser.add_argument("--res_attention", type=bool, default=None)
    parser.add_argument("--pre_norm", type=bool, default=None)
    parser.add_argument("--store_attn", type=bool, default=None)
    parser.add_argument(
        "--pe",
        type=str,
        default=None,
        choices=[
            "zeros",
            "zero",
            "normal",
            "gauss",
            "uniform",
            "lin1d",
            "exp1d",
            "lin2d",
            "exp2d",
            "sincos",
        ],
    )
    parser.add_argument("--learn_pe", type=bool, default=None)
    parser.add_argument("--pretrain_head", type=bool, default=None)
    parser.add_argument("--head_type", type=str, default=None, choices=["flatten"])
    parser.add_argument("--verbose", type=bool, default=None)


class PatchTST(nn.Module):
    """
    Paper: https://arxiv.org/abs/2211.14730
    Source: https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/models/PatchTST.py
    """

    def __init__(self, configs):
        super().__init__()
        backbone_config = {
            "c_in": configs.input_channels,
            "context_window": configs.input_len,
            "target_window": configs.output_len,
            "patch_len": configs.patch_len,
            "stride": configs.stride,
            "max_seq_len": configs.max_seq_len,
            "n_layers": configs.e_layers,
            "d_model": configs.d_model,
            "n_heads": configs.n_heads,
            "d_k": getattr(configs, "d_k", None),
            "d_v": getattr(configs, "d_v", None),
            "d_ff": configs.d_ff,
            "norm": configs.norm,
            "attn_dropout": configs.attn_dropout,
            "dropout": configs.dropout,
            "act": configs.act,
            "key_padding_mask": configs.key_padding_mask,
            "padding_var": getattr(configs, "padding_var", None),
            "attn_mask": getattr(configs, "attn_mask", None),
            "res_attention": configs.res_attention,
            "pre_norm": configs.pre_norm,
            "store_attn": configs.store_attn,
            "pe": configs.pe,
            "learn_pe": configs.learn_pe,
            "fc_dropout": configs.fc_dropout,
            "head_dropout": configs.head_dropout,
            "padding_patch": configs.padding_patch,
            "pretrain_head": configs.pretrain_head,
            "head_type": configs.head_type,
            "individual": configs.individual,
            "revin": configs.revin,
            "affine": configs.affine,
            "subtract_last": configs.subtract_last,
            "verbose": configs.verbose,
        }

        # Model initialization
        self.decomposition = configs.decomposition
        if self.decomposition:
            self.decomp_module = SeriesDecompMA(configs.kernel_size)
            self.model_trend = PatchTST_backbone(**backbone_config)
            self.model_res = PatchTST_backbone(**backbone_config)
        else:
            self.model = PatchTST_backbone(**backbone_config)

    def forward(self, x):  # x: [Batch, Input length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            # x: [Batch, Channel, Input length]
            res_init = res_init.permute(0, 2, 1)
            trend_init = trend_init.permute(0, 2, 1)
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0, 2, 1)  # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]
        return x


class PatchTST_backbone(nn.Module):
    def __init__(
        self,
        c_in: int,
        context_window: int,
        target_window: int,
        patch_len: int,
        stride: int,
        max_seq_len: Optional[int] = 1024,
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
        key_padding_mask: bool = "auto",
        padding_var: Optional[int] = None,
        attn_mask: Optional[Tensor] = None,
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        fc_dropout: float = 0.0,
        head_dropout=0,
        padding_patch=None,
        pretrain_head: bool = False,
        head_type="flatten",
        individual=False,
        revin=True,
        affine=True,
        subtract_last=False,
        verbose: bool = False,
        **kwargs,
    ):

        super().__init__()

        # RevIn
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == "end":  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        # Backbone
        self.backbone = TSTiEncoder(
            c_in,
            patch_num=patch_num,
            patch_len=patch_len,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            attn_dropout=attn_dropout,
            dropout=dropout,
            act=act,
            key_padding_mask=key_padding_mask,
            padding_var=padding_var,
            attn_mask=attn_mask,
            res_attention=res_attention,
            pre_norm=pre_norm,
            store_attn=store_attn,
            pe=pe,
            learn_pe=learn_pe,
            verbose=verbose,
            **kwargs,
        )

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head:
            # custom head passed as a partial func with all its kwargs
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout)
        elif head_type == "flatten":
            self.head = Flatten_Head(
                self.individual,
                self.n_vars,
                self.head_nf,
                target_window,
                head_dropout=head_dropout,
            )

    def forward(self, z):  # z: [bs x nvars x seq_len]
        # norm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, "norm")
            z = z.permute(0, 2, 1)

        # do patching
        if self.padding_patch == "end":
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0, 1, 3, 2)
        # z: [bs x nvars x patch_len x patch_num]

        # model
        z = self.backbone(z)  # z: [bs x nvars x d_model x patch_num]
        z = self.head(z)  # z: [bs x nvars x target_window]

        # denorm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, "denorm")
            z = z.permute(0, 2, 1)
        return z

    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout), nn.Conv1d(head_nf, vars, 1))


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x
