import torch
import torch.nn as nn

optional = {
    "d_model": 512,
    "d_ff": 2048,
    "fac_T": False,
    "fac_C": False,
    "sampling": 2,
    "norm": True,
    "e_layers": 2,
    "individual": False,
}


def args_update(parser):
    parser.add_argument("--d_model", type=int, default=None, help="dimension of model")
    parser.add_argument("--d_ff", type=int, default=None, help="dimension of fcn")
    parser.add_argument(
        "--fac_T",
        action="store_true",
        default=None,
        help="whether to apply factorized temporal interaction",
    )
    parser.add_argument(
        "--fac_C",
        action="store_true",
        default=None,
        help="whether to apply factorized channel interaction",
    )
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
    parser.add_argument("--individual", action="store_true", default=None)


class MTSMixer(nn.Module):
    """
    Paper: https://arxiv.org/abs/2302.04501
    Source: https://github.com/plumprc/MTS-Mixers/blob/main/models/MTSMixer.py
    """

    def __init__(self, configs):
        super().__init__()
        self.mlp_blocks = nn.ModuleList(
            [
                MixerBlock(
                    tokens_dim=configs.input_len,
                    channels_dim=configs.input_channels,
                    tokens_hidden_dim=configs.d_model,
                    channels_hidden_dim=configs.d_ff,
                    fac_T=configs.fac_T,
                    fac_C=configs.fac_C,
                    sampling=configs.sampling,
                    norm_flag=configs.norm,
                )
                for _ in range(configs.e_layers)
            ]
        )
        self.norm = nn.LayerNorm(configs.input_channels) if configs.norm else None
        self.projection = ChannelProjection(
            seq_len=configs.input_len,
            pred_len=configs.output_len,
            num_channel=configs.input_channels,
            individual=configs.individual,
        )

    def forward(self, x):
        for block in self.mlp_blocks:
            x = block(x)
        x = self.norm(x) if self.norm else x
        x = self.projection(x)
        return x


class ChannelProjection(nn.Module):
    def __init__(self, seq_len, pred_len, num_channel, individual):
        super().__init__()

        self.linears = (
            nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(num_channel)])
            if individual
            else nn.Linear(seq_len, pred_len)
        )
        self.individual = individual

    def forward(self, x):
        # x: [B, L, D]
        x_out = []
        if self.individual:
            for idx in range(x.shape[-1]):
                x_out.append(self.linears[idx](x[:, :, idx]))

            x = torch.stack(x_out, dim=-1)

        else:
            x = self.linears(x.transpose(1, 2)).transpose(1, 2)

        return x


class MLPBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, input_dim)

    def forward(self, x):
        # [B, L, D] or [B, D, L]
        return self.fc2(self.gelu(self.fc1(x)))


class FactorizedTemporalMixing(nn.Module):
    def __init__(self, input_dim, mlp_dim, sampling):
        super().__init__()

        assert sampling in [1, 2, 3, 4, 6, 8, 12]
        self.sampling = sampling
        self.temporal_fac = nn.ModuleList(
            [MLPBlock(input_dim // sampling, mlp_dim) for _ in range(sampling)]
        )

    def merge(self, shape, x_list):
        y = torch.zeros(shape, device=x_list[0].device)
        for idx, x_pad in enumerate(x_list):
            y[:, :, idx :: self.sampling] = x_pad

        return y

    def forward(self, x):
        x_samp = []
        for idx, samp in enumerate(self.temporal_fac):
            x_samp.append(samp(x[:, :, idx :: self.sampling]))

        x = self.merge(x.shape, x_samp)

        return x


class FactorizedChannelMixing(nn.Module):
    def __init__(self, input_dim, factorized_dim):
        super().__init__()

        assert input_dim > factorized_dim
        self.channel_mixing = MLPBlock(input_dim, factorized_dim)

    def forward(self, x):
        return self.channel_mixing(x)


class MixerBlock(nn.Module):
    def __init__(
        self,
        tokens_dim,
        channels_dim,
        tokens_hidden_dim,
        channels_hidden_dim,
        fac_T,
        fac_C,
        sampling,
        norm_flag,
    ):
        super().__init__()
        self.tokens_mixing = (
            FactorizedTemporalMixing(tokens_dim, tokens_hidden_dim, sampling)
            if fac_T
            else MLPBlock(tokens_dim, tokens_hidden_dim)
        )
        self.channels_mixing = (
            FactorizedChannelMixing(channels_dim, channels_hidden_dim)
            if fac_C
            else None
        )
        self.norm = nn.LayerNorm(channels_dim) if norm_flag else None

    def forward(self, x):
        # token-mixing [B, D, #tokens]
        y = self.norm(x) if self.norm else x
        y = self.tokens_mixing(y.transpose(1, 2)).transpose(1, 2)

        # channel-mixing [B, #tokens, D]
        if self.channels_mixing:
            y += x
            res = y
            y = self.norm(y) if self.norm else y
            y = res + self.channels_mixing(y)

        return y
