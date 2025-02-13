import torch
import torch.nn as nn

from layers import RevIN, SeriesDecompMA

optional = {
    "moving_avg": 25,
    "stride": 1,
    "hidden_size": 512,
    "noSCI": True,
}


def args_update(parser):
    parser.add_argument("--moving_avg", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--noSCI", type=bool, default=None)


class Amplifier(nn.Module):
    """
    Paper: https://arxiv.org/abs/2501.17216
    Source: https://github.com/aikunyi/Amplifier/blob/main/models/Amplifier.py
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.input_len
        self.pred_len = configs.output_len
        self.channels = configs.input_channels
        self.hidden_size = configs.hidden_size
        self.noSCI = configs.noSCI

        self.revin_layer = RevIN(
            num_features=self.channels, affine=True, subtract_last=False
        )

        self.decompsition = SeriesDecompMA(
            kernel_size=configs.moving_avg, stride=configs.stride
        )

        self.mask_matrix = nn.Parameter(
            torch.ones(int(self.seq_len / 2) + 1, self.channels)
        )
        self.freq_linear = nn.Linear(
            int(self.seq_len / 2) + 1, int(self.pred_len / 2) + 1
        ).to(torch.cfloat)

        self.linear_seasonal = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len),
        )

        self.linear_trend = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len),
        )

        if self.noSCI is False:
            # SCI block
            self.extract_common_pattern = nn.Sequential(
                nn.Linear(self.channels, self.channels),
                nn.LeakyReLU(),
                nn.Linear(self.channels, 1),
            )

            self.model_common_pattern = nn.Sequential(
                nn.Linear(self.seq_len, self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size, self.seq_len),
            )

            self.model_spacific_pattern = nn.Sequential(
                nn.Linear(self.seq_len, self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size, self.seq_len),
            )

    def forward(self, x):
        B, T, C = x.size()

        # RevIN
        z = x
        z = self.revin_layer(z, "norm")
        x = z

        # Energy Amplification Block
        x_fft = torch.fft.rfft(x, dim=1)  # domain conversion
        x_inverse_fft = torch.flip(x_fft, dims=[1])  # flip the spectrum
        x_inverse_fft = x_inverse_fft * self.mask_matrix
        x_amplifier_fft = x_fft + x_inverse_fft
        x_amplifier = torch.fft.irfft(x_amplifier_fft, dim=1)

        # SCI block
        if self.noSCI is False:
            x = x_amplifier
            # extract common pattern
            common_pattern = self.extract_common_pattern(x)
            common_pattern = self.model_common_pattern(
                common_pattern.permute(0, 2, 1)
            ).permute(0, 2, 1)
            # model specific pattern
            specififc_pattern = x - common_pattern.repeat(1, 1, C)
            specififc_pattern = self.model_spacific_pattern(
                specififc_pattern.permute(0, 2, 1)
            ).permute(0, 2, 1)

            x = specififc_pattern + common_pattern.repeat(1, 1, C)
            x_amplifier = x

        # Seasonal Trend Forecaster
        seasonal, trend = self.decompsition(x_amplifier)
        seasonal = self.linear_seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
        trend = self.linear_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
        out_amplifier = seasonal + trend

        # Energy Restoration Block
        out_amplifier_fft = torch.fft.rfft(out_amplifier, dim=1)
        x_inverse_fft = self.freq_linear(x_inverse_fft.permute(0, 2, 1)).permute(
            0, 2, 1
        )
        out_fft = out_amplifier_fft - x_inverse_fft
        out = torch.fft.irfft(out_fft, dim=1)

        # inverse RevIN
        z = out
        z = self.revin_layer(z, "denorm")
        out = z

        return out
