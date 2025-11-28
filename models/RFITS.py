import torch
import torch.nn as nn

optional = {"cut_freq": 0, "base_T": 24}


def args_update(parser):
    parser.add_argument("--cut_freq", type=int, default=None)
    parser.add_argument("--base_T", type=int, default=None)


class RFITS(nn.Module):
    """
    Paper: https://arxiv.org/abs/2307.03756
    Source: https://github.com/VEWOXIC/FITS/blob/main/models/Real_FITS.py
    FITS: Frequency Interpolation Time Series Forecasting
    This is the real value implementation of the original FITS.
    Real_FITS simulates the complex value multiplication with two layer of real value linear layer following
    Y_real = X_real*W_real - X_imag * W_imag
    Y_imag = X_real*W_imag + X_imag * W_real
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.input_len
        self.pred_len = configs.output_len
        self.channels = configs.input_channels

        self.cut_freq = (
            configs.cut_freq
            if configs.cut_freq != 0
            else self.pred_len // configs.base_T
        )
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len

        # Complex layer for frequency upcampling
        self.freq_upsampler_real = nn.Linear(
            in_features=self.cut_freq,
            out_features=int(self.cut_freq * self.length_ratio),
        )
        # Complex layer for frequency upcampling
        self.freq_upsampler_imag = nn.Linear(
            in_features=self.cut_freq,
            out_features=int(self.cut_freq * self.length_ratio),
        )

    def forward(self, x):
        # RevIN
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5

        x = x / torch.sqrt(x_var)

        # if self.training:
        low_specx = torch.fft.rfft(x, dim=1)

        low_specx = torch.view_as_real(low_specx[:, 0 : self.cut_freq, :])
        low_specx_real = low_specx[:, :, :, 0]
        low_specx_imag = low_specx[:, :, :, 1]

        # compute real/imag upsample inputs
        # low_specx_real/imag: [B, cut_freq, C]
        low_specx_real = low_specx_real  # already assigned above
        low_specx_imag = low_specx_imag  # already assigned above

        # prepare for linear upsampling: Linear expects [B, C, cut_freq] -> we permute to [B, C, cut_freq]
        inp_real = low_specx_real.permute(0, 2, 1)  # [B, C, cut_freq]
        inp_imag = low_specx_imag.permute(0, 2, 1)  # [B, C, cut_freq]

        # apply frequency upsamplers (produce [B, C, cut_freq_out])
        up_real_from_real = self.freq_upsampler_real(inp_real)
        up_real_from_imag = self.freq_upsampler_imag(inp_imag)
        up_imag_from_imag = self.freq_upsampler_real(inp_imag)
        up_imag_from_real = self.freq_upsampler_imag(inp_real)

        # bring back to shape [B, cut_freq_out, C]
        up_real_from_real = up_real_from_real.permute(0, 2, 1)
        up_real_from_imag = up_real_from_imag.permute(0, 2, 1)
        up_imag_from_imag = up_imag_from_imag.permute(0, 2, 1)
        up_imag_from_real = up_imag_from_real.permute(0, 2, 1)

        # combine to form complex-valued upsampled spectrum (real and imag parts)
        # Y_real = X_real*W_real - X_imag * W_imag
        low_specxy_real = up_real_from_real - up_real_from_imag

        # Y_imag = X_real*W_imag + X_imag * W_real
        low_specxy_imag = up_imag_from_imag + up_imag_from_real

        # prepare target frequency buffer size (rfft length for seq_len+pred_len)
        total_len = self.seq_len + self.pred_len
        target_freq_len = int(total_len / 2 + 1)

        # allocate zero buffers and copy computed low-frequency content in front
        B = low_specxy_real.size(0)
        C = low_specxy_real.size(2)
        freq_real_buf = torch.zeros(
            (B, target_freq_len, C),
            dtype=low_specxy_real.dtype,
            device=low_specxy_real.device,
        )
        freq_imag_buf = torch.zeros(
            (B, target_freq_len, C),
            dtype=low_specxy_imag.dtype,
            device=low_specxy_imag.device,
        )

        # place upsampled low-frequency coefficients at the start
        freq_real_buf[:, : low_specxy_real.size(1), :] = low_specxy_real
        freq_imag_buf[:, : low_specxy_imag.size(1), :] = low_specxy_imag

        # reconstruct complex spectrum, inverse FFT to time domain
        low_specxy = torch.complex(
            freq_real_buf, freq_imag_buf
        )  # [B, target_freq_len, C]
        low_xy = torch.fft.irfft(low_specxy, dim=1)  # [B, total_len, C]

        # Compensate the length change and undo RevIN normalization
        low_xy = low_xy * self.length_ratio
        xy = (low_xy) * torch.sqrt(x_var) + x_mean

        # return only the predicted future window (pred_len)
        start = self.seq_len
        end = start + self.pred_len
        pred = xy[:, start:end, :].contiguous()
        return pred
