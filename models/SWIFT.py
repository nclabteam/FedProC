import math

import torch
import torch.nn as nn
from pytorch_wavelets import DWT1DForward, DWT1DInverse

optional = {
    "conv_kernel": 3,
    "conv_dropout": 0.1,
    "fc_dropout": 0.05,
    "wav": "haar",
    "J": 1,
    "hidden_size": 128,
    "core": "Linear",
}


def args_update(parser):
    parser.add_argument(
        "--conv_kernel", type=int, default=None, help="convolution kernel size"
    )
    parser.add_argument(
        "--conv_dropout", type=float, default=None, help="convolution dropout"
    )
    parser.add_argument(
        "--fc_dropout", type=float, default=None, help="fully connected dropout"
    )
    parser.add_argument(
        "--wav", type=str, default=None, choices=wavelet_filter_lengths.keys()
    )
    parser.add_argument("--J", type=int, default=None)
    parser.add_argument(
        "--hidden_size", type=int, default=None, help="hidden_size of linear layer"
    )
    parser.add_argument("--core", type=str, default=None, choices=["Linear", "MLP"])


wavelet_filter_lengths = {
    "haar": 2,
    "db1": 2,
    "db2": 4,
    "db3": 6,
    "db4": 8,
    "db5": 10,
    "db6": 12,
    "db7": 14,
    "db8": 16,
    "db9": 18,
    "db10": 20,
    "sym2": 4,
    "sym3": 6,
    "sym4": 8,
    "sym5": 10,
    "sym6": 12,
    "sym7": 14,
    "sym8": 16,
    "coif1": 6,
    "coif2": 12,
    "coif3": 18,
    "coif4": 24,
    "coif5": 30,
    "bior1.1": 2,  # 2, 2 (decomposition and reconstruction filters)
    "bior2.2": 6,  # 6, 2 (decomposition and reconstruction filters)
    "bior3.3": 10,  # 10, 2 (decomposition and reconstruction filters)
    "bior4.4": 14,  # 14, 2 (decomposition and reconstruction filters)
    "rbio1.1": 2,  # 2, 2 (reverse biorthogonal)
    "rbio2.2": 6,  # 6, 2 (reverse biorthogonal)
    "rbio3.3": 10,  # 10, 2 (reverse biorthogonal)
    "rbio4.4": 14,  # 14, 2 (reverse biorthogonal)
    "dmey": 102,  # Discrete Meyer wavelet
    "gaus1": 2,
    "gaus2": 4,
    "gaus3": 6,
    "mexh": "N/A",  # Mexican Hat wavelet, depends on scale
    "morl": "N/A",  # Morlet wavelet, depends on scale
}


def compute_dwt_dimensions(T, J, wav):
    filter_length = wavelet_filter_lengths[wav]
    P = filter_length - 1
    yh_lengths = []
    for _ in range(1, J + 1):
        T = math.floor((T + P) / 2)
        yh_lengths.append(T)
    yl_length = T
    return yl_length, yh_lengths


class SWIFT(nn.Module):
    """
    Paper: https://arxiv.org/abs/2501.16178
    Source: https://github.com/LancelotXWX/SWIFT/blob/main/models/SWIFT_Linear.py
    """

    def __init__(self, configs):
        super().__init__()

        self.seq_len = configs.input_len
        self.pred_len = configs.output_len
        self.enc_in = configs.input_channels
        self.conv_kernel = configs.conv_kernel
        self.conv_dropout = configs.conv_dropout
        self.fc_dropout = configs.fc_dropout
        self.wav = configs.wav
        self.J = configs.J
        self.hidden_size = configs.hidden_size

        self.conv = nn.Conv1d(
            in_channels=2,
            out_channels=2,
            kernel_size=self.conv_kernel,
            stride=1,
            padding=(self.conv_kernel - 1) // 2,
            padding_mode="zeros",
            bias=False,
        )
        self.dwt = DWT1DForward(wave=self.wav, J=self.J)
        self.idwt = DWT1DInverse(wave=self.wav)

        yl, _ = compute_dwt_dimensions(T=self.seq_len, J=self.J, wav=self.wav)
        yl_, _ = compute_dwt_dimensions(T=self.pred_len, J=self.J, wav=self.wav)

        if configs.core == "Linear":
            self.yl_upsampler = nn.Linear(yl, yl_)
        elif configs.core == "MLP":
            self.yl_upsampler = nn.Sequential(
                nn.Linear(yl, self.hidden_size),
                nn.ELU(),
                nn.Linear(self.hidden_size, yl_),
            )

        self.dropout1 = nn.Dropout(self.conv_dropout)
        self.dropout2 = nn.Dropout(self.fc_dropout)

    def forward(self, x):
        # normalization and permute     b,s,c -> b,c,s
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)

        # DWT
        yl, yh = self.dwt(x)
        yh = yh[0]
        # yh = self.dropout1(self.conv(yh))
        y = torch.stack([yl, yh], dim=-2)
        y = (
            self.conv(y.reshape(int(y.size(0) * self.enc_in), 2, y.size(-1))).reshape(
                -1, self.enc_in, 2, y.size(-1)
            )
            + y
        )
        y = self.dropout1(y)

        # Up sample
        y = self.yl_upsampler(y)
        yl_, yh_ = y[:, :, 0, :], [y[:, :, 1, :]]

        # IDWT
        y = self.idwt((yl_, yh_))

        y = y.permute(0, 2, 1) + seq_mean

        return y
