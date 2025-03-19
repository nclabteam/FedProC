import torch
import torch.nn as nn

optional = {
    "sampling": 2,
    "norm": True,
    "e_layers": 2,
    "mat": "random",
}


def args_update(parser):
    parser.add_argument(
        "--sampling",
        type=int,
        default=None,
        help="the number of downsampling in factorized temporal interaction",
    )
    parser.add_argument(
        "--norm", action="store_false", default=None, help="whether to apply LayerNorm"
    )
    parser.add_argument(
        "--e_layers", type=int, default=None, help="num of encoder layers"
    )
    parser.add_argument("--mat", type=str, default=None, choices=["random", "identity"])


class MTSMatrix(nn.Module):
    """
    Paper: https://arxiv.org/abs/2302.04501
    Source: https://github.com/plumprc/MTS-Mixers/blob/main/models/MTSMatrix.py
    """

    def __init__(self, configs):
        super().__init__()
        self.matrics = nn.ModuleList(
            [
                FactorizedTemporalMixing(
                    seq_len=configs.input_len,
                    enc_in=configs.input_channels,
                    sampling=configs.sampling,
                    mat=configs.mat,
                    norm=configs.norm,
                )
                for _ in range(configs.e_layers)
            ]
        )
        self.projection = nn.Linear(configs.input_len, configs.output_len)

    def forward(self, x):
        for mat in self.matrics:
            x = mat(x)
        x = self.projection(x.transpose(1, 2)).transpose(1, 2)
        return x


class Matrix(nn.Module):
    def __init__(self, seq_len, enc_in, mat, norm):
        super().__init__()
        self.temporal = (
            nn.Parameter(torch.rand(seq_len, seq_len))
            if mat == "random"
            else nn.Parameter(torch.eye(seq_len))
        )
        self.channels = (
            nn.Parameter(torch.rand(enc_in, enc_in))
            if mat == "random"
            else nn.Parameter(torch.eye(enc_in, enc_in))
        )
        self.norm = nn.LayerNorm(enc_in)
        self.acti = nn.GELU()
        self.is_norm = norm

    def forward(self, x):
        x = x + self.acti(torch.matmul(self.temporal, x))
        x = (
            x + self.norm(torch.matmul(x, self.channels))
            if self.is_norm
            else x + torch.matmul(x, self.channels)
        )

        return x


class FactorizedTemporalMixing(nn.Module):
    def __init__(self, seq_len, enc_in, sampling, mat, norm):
        super().__init__()

        assert sampling in [1, 2, 3, 4, 6, 8, 12]
        self.sampling = sampling
        self.temporal_fac = nn.ModuleList(
            [Matrix(seq_len // sampling, enc_in, mat, norm) for _ in range(sampling)]
        )

    def merge(self, shape, x_list):
        y = torch.zeros(shape, device=x_list[0].device)
        for idx, x_pad in enumerate(x_list):
            y[:, idx :: self.sampling, :] = x_pad

        return y

    def forward(self, x):
        x_samp = []
        for idx, samp in enumerate(self.temporal_fac):
            x_samp.append(samp(x[:, idx :: self.sampling, :]))

        x = self.merge(x.shape, x_samp)

        return x
