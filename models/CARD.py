import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from layers import Transpose

optional = {
    "patch_len": 16,
    "stride": 8,
    "d_model": 16,
    "dropout": 0.3,
    "use_statistic": False,
    "e_layers": 2,
    "momentum": 0.1,
    "n_heads": 2,
    "d_ff": 32,
    "dp_rank": 8,
    "alpha": 0.5,
    "merge_size": 2,
}


def args_update(parser):
    parser.add_argument("--patch_len", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--use_statistic", type=bool, default=None)
    parser.add_argument("--e_layers", type=int, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--dp_rank", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--merge_size", type=int, default=None)


class CARD(nn.Module):
    """
    Paper: https://arxiv.org/abs/2305.12095
    Source: https://github.com/wxie9/CARD/blob/main/long_term_forecast_l720/models/CARD.py
    """

    def __init__(self, configs):
        super().__init__()
        self.model = CARDformer(
            patch_len=configs.patch_len,
            stride=configs.stride,
            d_model=configs.d_model,
            patch_num=int((configs.input_len - configs.patch_len) / configs.stride + 1),
            enc_in=configs.input_channels,
            dropout=configs.dropout,
            use_statistic=configs.use_statistic,
            pred_len=configs.output_len,
            e_layers=configs.e_layers,
            momentum=configs.momentum,
            n_heads=configs.n_heads,
            dp_rank=configs.dp_rank,
            d_ff=configs.d_ff,
            merge_size=configs.merge_size,
            alpha=configs.alpha,
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.model(x)
        x = x.permute(0, 2, 1)
        return x


class CARDformer(nn.Module):
    def __init__(
        self,
        patch_len: int,
        stride: int,
        d_model: int,
        patch_num: int,
        enc_in: int,
        dropout: float,
        pred_len: int,
        e_layers: int,
        momentum: float,
        dp_rank,
        d_ff: int,
        merge_size: int,
        alpha: float,
        n_heads: int,
        use_statistic: bool = False,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.W_pos_embed = nn.Parameter(torch.randn(patch_num, d_model) * 1e-2)
        self.model_token_number = 0
        if self.model_token_number > 0:
            self.model_token = nn.Parameter(
                torch.randn(enc_in, self.model_token_number, d_model) * 1e-2
            )
        self.W_input_projection = nn.Linear(self.patch_len, d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.use_statistic = use_statistic
        self.W_statistic = nn.Linear(2, d_model)
        self.cls = nn.Parameter(torch.randn(1, d_model) * 1e-2)

        self.W_out = nn.Linear(
            (patch_num + 1 + self.model_token_number) * d_model, pred_len
        )
        total_token_number = patch_num + self.model_token_number + 1
        self.Attentions_over_token = nn.ModuleList(
            [
                Attention(
                    n_heads=n_heads,
                    enc_in=enc_in,
                    d_model=d_model,
                    dropout=dropout,
                    momentum=momentum,
                    dp_rank=dp_rank,
                    d_ff=d_ff,
                    merge_size=merge_size,
                    total_token_number=total_token_number,
                    alpha=alpha,
                    over_hidden=False,
                )
                for _ in range(e_layers)
            ]
        )
        self.Attentions_over_channel = nn.ModuleList(
            [
                Attention(
                    n_heads=n_heads,
                    enc_in=enc_in,
                    d_model=d_model,
                    dropout=dropout,
                    momentum=momentum,
                    dp_rank=dp_rank,
                    d_ff=d_ff,
                    merge_size=merge_size,
                    total_token_number=total_token_number,
                    alpha=alpha,
                    over_hidden=True,
                )
                for _ in range(e_layers)
            ]
        )
        self.Attentions_mlp = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(e_layers)]
        )
        self.Attentions_dropout = nn.ModuleList(
            [nn.Dropout(dropout) for i in range(e_layers)]
        )
        self.Attentions_norm = nn.ModuleList(
            [
                nn.Sequential(
                    Transpose(1, 2),
                    nn.BatchNorm1d(d_model, momentum=momentum),
                    Transpose(1, 2),
                )
                for i in range(e_layers)
            ]
        )

    def forward(self, z):
        b, c, s = z.shape

        z_mean = torch.mean(z, dim=(-1), keepdims=True)
        z_std = torch.std(z, dim=(-1), keepdims=True)
        z = (z - z_mean) / (z_std + 1e-4)

        zcube = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z_embed = self.input_dropout(self.W_input_projection(zcube)) + self.W_pos_embed

        if self.use_statistic:
            z_stat = torch.cat((z_mean, z_std), dim=-1)
            if z_stat.shape[-2] > 1:
                z_stat = (z_stat - torch.mean(z_stat, dim=-2, keepdims=True)) / (
                    torch.std(z_stat, dim=-2, keepdims=True) + 1e-4
                )
            z_stat = self.W_statistic(z_stat)
            z_embed = torch.cat((z_stat.unsqueeze(-2), z_embed), dim=-2)
        else:
            cls_token = self.cls.repeat(z_embed.shape[0], z_embed.shape[1], 1, 1)
            z_embed = torch.cat((cls_token, z_embed), dim=-2)

        inputs = z_embed
        b, c, t, h = inputs.shape
        for a_2, a_1, mlp, drop, norm in zip(
            self.Attentions_over_token,
            self.Attentions_over_channel,
            self.Attentions_mlp,
            self.Attentions_dropout,
            self.Attentions_norm,
        ):
            output_1 = a_1(inputs.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
            output_2 = a_2(output_1)
            outputs = drop(mlp(output_1 + output_2)) + inputs
            outputs = norm(outputs.reshape(b * c, t, -1)).reshape(b, c, t, -1)
            inputs = outputs

        z_out = self.W_out(outputs.reshape(b, c, -1))
        z = z_out * (z_std + 1e-4) + z_mean
        return z


class Attention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        enc_in: int,
        d_model: int,
        dropout: float,
        momentum: float,
        dp_rank: float,
        d_ff: int,
        merge_size: int,
        total_token_number: int,
        alpha: float,
        over_hidden: bool = False,
    ):
        super().__init__()

        self.over_hidden = over_hidden
        self.n_heads = n_heads
        self.c_in = enc_in
        self.qkv = nn.Linear(d_model, d_model * 3, bias=True)

        self.attn_dropout = nn.Dropout(dropout)
        self.head_dim = d_model // n_heads

        self.norm_post1 = nn.Sequential(
            Transpose(1, 2),
            nn.BatchNorm1d(d_model, momentum=momentum),
            Transpose(1, 2),
        )
        self.norm_post2 = nn.Sequential(
            Transpose(1, 2),
            nn.BatchNorm1d(d_model, momentum=momentum),
            Transpose(1, 2),
        )

        self.norm_attn = nn.Sequential(
            Transpose(1, 2),
            nn.BatchNorm1d(d_model, momentum=momentum),
            Transpose(1, 2),
        )

        self.dp_rank = dp_rank
        self.dp_k = nn.Linear(self.head_dim, self.dp_rank)
        self.dp_v = nn.Linear(self.head_dim, self.dp_rank)

        self.ff_1 = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=True),
        )

        self.ff_2 = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=True),
        )
        self.merge_size = merge_size

        ema_size = max(enc_in, total_token_number, dp_rank)
        ema_matrix = torch.zeros((ema_size, ema_size))
        ema_matrix[0][0] = 1
        for i in range(1, total_token_number):
            for j in range(i):
                ema_matrix[i][j] = ema_matrix[i - 1][j] * (1 - alpha)
            ema_matrix[i][i] = alpha
        self.register_buffer("ema_matrix", ema_matrix)

    def ema(self, src):
        return torch.einsum(
            "bnhad,ga ->bnhgd", src, self.ema_matrix[: src.shape[-2], : src.shape[-2]]
        )

    def dynamic_projection(self, src, mlp):
        src_dp = mlp(src)
        src_dp = F.softmax(src_dp, dim=-1)
        src_dp = torch.einsum("bnhef,bnhec -> bnhcf", src, src_dp)
        return src_dp

    def forward(self, src):
        (B, nvars, H, C) = src.shape

        qkv = (
            self.qkv(src)
            .reshape(B, nvars, H, 3, self.n_heads, C // self.n_heads)
            .permute(3, 0, 1, 4, 2, 5)
        )

        q, k, v = qkv[0], qkv[1], qkv[2]

        if not self.over_hidden:
            attn_score_along_token = (
                torch.einsum("bnhed,bnhfd->bnhef", self.ema(q), self.ema(k))
                / self.head_dim**-0.5
            )

            attn_along_token = self.attn_dropout(
                F.softmax(attn_score_along_token, dim=-1)
            )

            output_along_token = torch.einsum("bnhef,bnhfd->bnhed", attn_along_token, v)
        else:
            v_dp = self.dynamic_projection(v, self.dp_v)
            k_dp = self.dynamic_projection(k, self.dp_k)
            attn_score_along_token = (
                torch.einsum("bnhed,bnhfd->bnhef", self.ema(q), self.ema(k_dp))
                / self.head_dim**-0.5
            )

            attn_along_token = self.attn_dropout(
                F.softmax(attn_score_along_token, dim=-1)
            )
            output_along_token = torch.einsum(
                "bnhef,bnhfd->bnhed", attn_along_token, v_dp
            )

        attn_score_along_hidden = (
            torch.einsum("bnhae,bnhaf->bnhef", q, k) / q.shape[-2] ** -0.5
        )
        attn_along_hidden = self.attn_dropout(
            F.softmax(attn_score_along_hidden, dim=-1)
        )
        output_along_hidden = torch.einsum("bnhef,bnhaf->bnhae", attn_along_hidden, v)

        merge_size = self.merge_size

        output1 = rearrange(
            output_along_token.reshape(B * nvars, -1, self.head_dim),
            "bn (hl1 hl2 hl3) d -> bn  hl2 (hl3 hl1) d",
            hl1=self.n_heads // merge_size,
            hl2=output_along_token.shape[-2],
            hl3=merge_size,
        ).reshape(B * nvars, -1, self.head_dim * self.n_heads)

        output2 = rearrange(
            output_along_hidden.reshape(B * nvars, -1, self.head_dim),
            "bn (hl1 hl2 hl3) d -> bn  hl2 (hl3 hl1) d",
            hl1=self.n_heads // merge_size,
            hl2=output_along_token.shape[-2],
            hl3=merge_size,
        ).reshape(B * nvars, -1, self.head_dim * self.n_heads)

        output1 = self.norm_post1(output1)
        output1 = output1.reshape(B, nvars, -1, self.n_heads * self.head_dim)
        output2 = self.norm_post2(output2)
        output2 = output2.reshape(B, nvars, -1, self.n_heads * self.head_dim)

        src2 = self.ff_1(output1) + self.ff_2(output2)

        src = src + src2
        src = src.reshape(B * nvars, -1, self.n_heads * self.head_dim)
        src = self.norm_attn(src)

        src = src.reshape(B, nvars, -1, self.n_heads * self.head_dim)
        return src
