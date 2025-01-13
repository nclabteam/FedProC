import math

import torch
import torch.nn as nn

from layers import SeriesDecompDEMA, SeriesDecompEMA

optional = {
    "alpha": 0.3,
    "beta": 0.3,
    "patch_len": 6,
    "stride": 3,
    "padding_patch": "end",
    "ma_type": "EMA",
    "learnable": False,
}


def args_update(parser):
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="smoothing factor for EMA or DEMA",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="smoothing factor for DEMA",
    )
    parser.add_argument("--patch_len", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument(
        "--padding_patch",
        type=str,
        default=None,
        choices=["end", "None"],
        help="None: None; end: padding on the end",
    )
    parser.add_argument(
        "--ma_type",
        type=str,
        default=None,
        choices=["EMA", "DEMA", "None"],
    )
    parser.add_argument(
        "--learnable",
        type=bool,
        default=None,
        help="learnable alpha and beta",
    )


class xPatch(nn.Module):
    def __init__(self, configs):
        super().__init__()
        if configs.ma_type == "EMA":
            self.decomp = SeriesDecompEMA(
                alpha=configs.alpha,
                learnable=configs.learnable,
                device=configs.device,
            )
        elif configs.ma_type == "DEMA":
            self.decomp = SeriesDecompDEMA(
                alpha=configs.alpha,
                beta=configs.beta,
                learnable=configs.learnable,
                device=configs.device,
            )
        elif configs.ma_type == "None":
            # No decomposition
            self.decomp = NoneDecomp()

        self.net = Network(
            seq_len=configs.input_len,
            pred_len=configs.output_len,
            patch_len=configs.patch_len,
            stride=configs.stride,
            padding_patch=configs.padding_patch,
        )

    def forward(self, x):
        # x: [Batch, Input, Channel]
        seasonal_init, trend_init = self.decomp(x)
        x = self.net(seasonal_init, trend_init)

        return x


class NoneDecomp(nn.Module):
    def forward(self, x):
        return x, x


class Network(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch):
        super().__init__()

        # Parameters
        self.pred_len = pred_len

        # Non-linear Stream
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = patch_len * patch_len
        self.patch_num = (seq_len - patch_len) // stride + 1
        if padding_patch == "end":  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1

        # Patch Embedding
        self.fc1 = nn.Linear(patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)

        # CNN Depthwise
        self.conv1 = nn.Conv1d(
            self.patch_num, self.patch_num, patch_len, patch_len, groups=self.patch_num
        )
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

        # Residual Stream
        self.fc2 = nn.Linear(self.dim, patch_len)

        # CNN Pointwise
        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)

        # Flatten Head
        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * patch_len, pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(pred_len * 2, pred_len)

        # Linear Stream
        # MLP
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)

        self.fc6 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)

        self.fc7 = nn.Linear(pred_len // 2, pred_len)

        # Streams Concatination
        self.fc8 = nn.Linear(pred_len * 2, pred_len)

    def forward(self, s, t):
        # x: [Batch, Input, Channel]
        # s - seasonality
        # t - trend

        s = s.permute(0, 2, 1)  # to [Batch, Channel, Input]
        t = t.permute(0, 2, 1)  # to [Batch, Channel, Input]

        # Channel split for channel independence
        B = s.shape[0]  # Batch size
        C = s.shape[1]  # Channel size
        I = s.shape[2]  # Input size
        s = torch.reshape(s, (B * C, I))  # [Batch and Channel, Input]
        t = torch.reshape(t, (B * C, I))  # [Batch and Channel, Input]

        # Non-linear Stream
        # Patching
        if self.padding_patch == "end":
            s = self.padding_patch_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # s: [Batch and Channel, Patch_num, Patch_len]

        # Patch Embedding
        s = self.fc1(s)
        s = self.gelu1(s)
        s = self.bn1(s)

        res = s

        # CNN Depthwise
        s = self.conv1(s)
        s = self.gelu2(s)
        s = self.bn2(s)

        # Residual Stream
        res = self.fc2(res)
        s = s + res

        # CNN Pointwise
        s = self.conv2(s)
        s = self.gelu3(s)
        s = self.bn3(s)

        # Flatten Head
        s = self.flatten1(s)
        s = self.fc3(s)
        s = self.gelu4(s)
        s = self.fc4(s)

        # Linear Stream
        # MLP
        t = self.fc5(t)
        t = self.avgpool1(t)
        t = self.ln1(t)

        t = self.fc6(t)
        t = self.avgpool2(t)
        t = self.ln2(t)

        t = self.fc7(t)

        # Streams Concatination
        x = torch.cat((s, t), dim=1)
        x = self.fc8(x)

        # Channel concatination
        x = torch.reshape(x, (B, C, self.pred_len))  # [Batch, Channel, Output]

        x = x.permute(0, 2, 1)  # to [Batch, Output, Channel]

        return x
