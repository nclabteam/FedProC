import torch
import torch.nn as nn

from layers import RevIN

optional = {
    "scale": 0.02,
    "hidden_size": 256,
}


def args_update(parser):
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)


class PaiFilter(nn.Module):
    """
    Paper: https://arxiv.org/abs/2411.01623
    Source: https://github.com/aikunyi/FilterNet/blob/main/models/PaiFilter.py
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.input_len
        self.pred_len = configs.output_len
        self.scale = configs.scale
        self.channels = configs.input_channels
        self.embed_size = self.seq_len
        self.hidden_size = configs.hidden_size

        self.revin_layer = RevIN(num_features=self.channels, affine=True)

        self.w = nn.Parameter(self.scale * torch.randn(1, self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len),
        )

    def circular_convolution(self, x, w):
        x = torch.fft.rfft(x, dim=2, norm="ortho")
        w = torch.fft.rfft(w, dim=1, norm="ortho")
        y = x * w
        out = torch.fft.irfft(y, n=self.embed_size, dim=2, norm="ortho")
        return out

    def forward(self, x):
        z = x
        z = self.revin_layer(z, "norm")
        x = z

        x = x.permute(0, 2, 1)

        x = self.circular_convolution(x, self.w.to(x.device))  # B, N, D

        x = self.fc(x)
        x = x.permute(0, 2, 1)

        z = x
        z = self.revin_layer(z, "denorm")
        x = z

        return x
