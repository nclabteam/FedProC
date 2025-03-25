import torch
import torch.nn as nn

from layers.DataEmbedding import DataEmbedding_wo_pos
from layers.SeriesDecompMA import SeriesDecompMA as series_decomp

optional = {
    "e_layers": 2,
    "down_sampling_layers": 1,
    "down_sampling_window": 2,
    "d_model": 16,
    "begin_order": 0,
    "use_norm": 1,
    "moving_avg": 25,
    "dropout": 0.1,
    "embed": "timeF",
}


def args_update(parser):
    parser.add_argument("--e_layers", type=int, default=None)
    parser.add_argument("--down_sampling_layers", type=int, default=None)
    parser.add_argument("--down_sampling_window", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--begin_order", type=int, default=None)
    parser.add_argument("--use_norm", type=int, default=None)
    parser.add_argument("--moving_avg", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--embed", type=str, default=None)


class TimeKAN(nn.Module):
    """
    Paper: https://arxiv.org/abs/2502.06910
    Source: https://github.com/huangst21/TimeKAN/blob/main/models/TimeKAN.py
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        self.res_blocks = nn.ModuleList(
            [FrequencyDecomp(configs) for _ in range(configs.e_layers)]
        )
        self.add_blocks = nn.ModuleList(
            [FrequencyMixing(configs) for _ in range(configs.e_layers)]
        )

        self.preprocess = series_decomp(configs.moving_avg)

        self.enc_embedding = DataEmbedding_wo_pos(
            1, configs.d_model, configs.embed, configs.granularity_unit, configs.dropout
        )
        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(
                    self.configs.input_channels,
                    affine=True,
                    non_norm=True if configs.use_norm == 0 else False,
                )
                for i in range(configs.down_sampling_layers + 1)
            ]
        )
        self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)
        self.predict_layer = nn.Linear(
            configs.input_len,
            configs.output_len,
        )

    def __multi_level_process_inputs(self, x_enc):
        down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)
        x_enc_ori = x_enc
        x_enc_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling
        x_enc = x_enc_sampling_list
        return x_enc

    def forward(self, x_enc):
        x_enc = self.__multi_level_process_inputs(x_enc)
        x_list = []
        for i, x in zip(
            range(len(x_enc)),
            x_enc,
        ):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, "norm")
            x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)

        enc_out_list = []
        for i, x in zip(range(len(x_list)), x_list):
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        for i in range(self.configs.e_layers):
            enc_out_list = self.res_blocks[i](enc_out_list)
            enc_out_list = self.add_blocks[i](enc_out_list)

        dec_out = enc_out_list[0]
        dec_out = self.predict_layer(dec_out.permute(0, 2, 1)).permute(0, 2, 1)
        dec_out = (
            self.projection_layer(dec_out)
            .reshape(B, self.configs.output_channels, self.configs.output_len)
            .permute(0, 2, 1)
            .contiguous()
        )
        dec_out = self.normalize_layers[0](dec_out, "denorm")
        return dec_out


class Normalize(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps=1e-5,
        affine=False,
        subtract_last=False,
        non_norm=False,
    ):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class ChebyKANLayer(nn.Module):
    def __init__(self, in_features, out_features, order):
        super().__init__()
        self.fc1 = ChebyKANLinear(in_features, out_features, order)

    def forward(self, x):
        B, N, C = x.shape
        x = self.fc1(x.reshape(B * N, C))
        x = x.reshape(B, N, -1).contiguous()
        return x


class FrequencyDecomp(nn.Module):
    def __init__(self, configs):
        super(FrequencyDecomp, self).__init__()
        self.configs = configs

    def forward(self, level_list):
        level_list_reverse = level_list.copy()
        level_list_reverse.reverse()
        out_low = level_list_reverse[0]
        out_high = level_list_reverse[1]
        out_level_list = [out_low]
        for i in range(len(level_list_reverse) - 1):
            out_high_res = self.frequency_interpolation(
                out_low.transpose(1, 2),
                self.configs.input_len
                // (
                    self.configs.down_sampling_window
                    ** (self.configs.down_sampling_layers - i)
                ),
                self.configs.input_len
                // (
                    self.configs.down_sampling_window
                    ** (self.configs.down_sampling_layers - i - 1)
                ),
            ).transpose(1, 2)
            out_high_left = out_high - out_high_res
            out_low = out_high
            if i + 2 <= len(level_list_reverse) - 1:
                out_high = level_list_reverse[i + 2]
            out_level_list.append(out_high_left)
        out_level_list.reverse()
        return out_level_list

    def frequency_interpolation(self, x, input_len, target_len):
        len_ratio = input_len / target_len
        x_fft = torch.fft.rfft(x, dim=2)
        out_fft = torch.zeros(
            [x_fft.size(0), x_fft.size(1), target_len // 2 + 1], dtype=x_fft.dtype
        ).to(x_fft.device)
        out_fft[:, :, : input_len // 2 + 1] = x_fft
        out = torch.fft.irfft(out_fft, dim=2)
        out = out * len_ratio
        return out


class FrequencyMixing(nn.Module):
    def __init__(self, configs):
        super(FrequencyMixing, self).__init__()
        self.configs = configs
        self.front_block = M_KAN(
            configs.d_model,
            self.configs.input_len
            // (
                self.configs.down_sampling_window ** (self.configs.down_sampling_layers)
            ),
            order=configs.begin_order,
        )

        self.front_blocks = torch.nn.ModuleList(
            [
                M_KAN(
                    configs.d_model,
                    self.configs.input_len
                    // (
                        self.configs.down_sampling_window
                        ** (self.configs.down_sampling_layers - i - 1)
                    ),
                    order=i + configs.begin_order + 1,
                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, level_list):
        level_list_reverse = level_list.copy()
        level_list_reverse.reverse()
        out_low = level_list_reverse[0]
        out_high = level_list_reverse[1]
        out_low = self.front_block(out_low)
        out_level_list = [out_low]
        for i in range(len(level_list_reverse) - 1):
            out_high = self.front_blocks[i](out_high)
            out_high_res = self.frequency_interpolation(
                out_low.transpose(1, 2),
                self.configs.input_len
                // (
                    self.configs.down_sampling_window
                    ** (self.configs.down_sampling_layers - i)
                ),
                self.configs.input_len
                // (
                    self.configs.down_sampling_window
                    ** (self.configs.down_sampling_layers - i - 1)
                ),
            ).transpose(1, 2)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(level_list_reverse) - 1:
                out_high = level_list_reverse[i + 2]
            out_level_list.append(out_low)
        out_level_list.reverse()
        return out_level_list

    def frequency_interpolation(self, x, input_len, target_len):
        len_ratio = input_len / target_len
        x_fft = torch.fft.rfft(x, dim=2)
        out_fft = torch.zeros(
            [x_fft.size(0), x_fft.size(1), target_len // 2 + 1], dtype=x_fft.dtype
        ).to(x_fft.device)
        out_fft[:, :, : input_len // 2 + 1] = x_fft
        out = torch.fft.irfft(out_fft, dim=2)
        out = out * len_ratio
        return out


class M_KAN(nn.Module):
    def __init__(self, d_model, input_len, order):
        super().__init__()
        self.channel_mixer = nn.Sequential(ChebyKANLayer(d_model, d_model, order))
        self.conv = BasicConv(
            d_model, d_model, kernel_size=3, degree=order, groups=d_model
        )

    def forward(self, x):
        x1 = self.channel_mixer(x)
        x2 = self.conv(x)
        out = x1 + x2
        return out


class BasicConv(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        kernel_size,
        degree,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        act=False,
        bn=False,
        bias=False,
        dropout=0.0,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = c_out
        self.conv = nn.Conv1d(
            c_in,
            c_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm1d(c_out) if bn else None
        self.act = nn.GELU() if act else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.bn is not None:
            x = self.bn(x)
        x = self.conv(x.transpose(-1, -2)).transpose(-1, -2)
        if self.act is not None:
            x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


# This is inspired by Kolmogorov-Arnold Networks but using Chebyshev polynomials instead of splines coefficients
class ChebyKANLinear(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLinear, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        self.epsilon = 1e-7
        self.pre_mul = False
        self.post_mul = False
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        # View and repeat input degree + 1 times
        b, c_in = x.shape
        if self.pre_mul:
            mul_1 = x[:, ::2]
            mul_2 = x[:, 1::2]
            mul_res = mul_1 * mul_2
            x = torch.concat([x[:, : x.shape[1] // 2], mul_res])
        x = x.view((b, c_in, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, inputdim, self.degree + 1)
        # Apply acos
        x = torch.tanh(x)
        x = torch.tanh(x)
        x = torch.acos(x)
        # x = torch.acos(torch.clamp(x, -1 + self.epsilon, 1 - self.epsilon))
        # # Multiply by arange [0 .. degree]
        x = x * self.arange
        # Apply cos
        x = x.cos()
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        if self.post_mul:
            mul_1 = y[:, ::2]
            mul_2 = y[:, 1::2]
            mul_res = mul_1 * mul_2
            y = torch.concat([y[:, : y.shape[1] // 2], mul_res])
        return y
