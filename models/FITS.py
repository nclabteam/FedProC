import torch
import torch.nn as nn

optional = {"cut_freq": 0, "base_T": 24}


def args_update(parser):
    parser.add_argument("--cut_freq", type=int, default=None)
    parser.add_argument("--base_T", type=int, default=None)


class FITS(nn.Module):
    """
    FITS: Frequency Interpolation Time Series Forecasting
    Paper: https://arxiv.org/abs/2307.03756
    Source: https://github.com/VEWOXIC/FITS/blob/main/models/FITS.py
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.input_len
        self.pred_len = configs.output_len
        self.channels = configs.input_channels

        self.dominance_freq = (
            configs.cut_freq
            if configs.cut_freq != 0
            else self.pred_len // configs.base_T
        )
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len

        # Complex layer for frequency upcampling
        self.freq_upsampler = nn.Linear(
            self.dominance_freq, int(self.dominance_freq * self.length_ratio)
        ).to(torch.cfloat)

    def forward(self, x):
        # RIN
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5

        x = x / torch.sqrt(x_var)

        low_specx = torch.fft.rfft(x, dim=1)
        low_specx[:, self.dominance_freq :] = 0  # LPF
        low_specx = low_specx[:, 0 : self.dominance_freq, :]  # LPF

        low_specxy_ = self.freq_upsampler(low_specx.permute(0, 2, 1)).permute(0, 2, 1)

        low_specxy = torch.zeros(
            [
                low_specxy_.size(0),
                int((self.seq_len + self.pred_len) / 2 + 1),
                low_specxy_.size(2),
            ],
            dtype=low_specxy_.dtype,
        ).to(low_specxy_.device)
        low_specxy[:, 0 : low_specxy_.size(1), :] = low_specxy_  # zero padding
        low_xy = torch.fft.irfft(low_specxy, dim=1)
        # energy compensation for the length change
        low_xy = low_xy * self.length_ratio

        # REVERSE RIN
        xy = (low_xy) * torch.sqrt(x_var) + x_mean

        # return only predicted future window to match training targets
        start = self.seq_len
        end = start + self.pred_len
        pred = xy[:, start:end, :].contiguous()
        return pred
