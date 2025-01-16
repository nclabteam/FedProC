import torch
import torch.nn as nn

from layers import DualAttention

optional = {
    "d_model": 512,
    "n_layers": 1,
    "dropout": 0,
    "positional_encoding_type": "sincos",
}


def update_args(parser):
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument(
        "--positional_encoding_type",
        type=str,
        default=None,
        choices=[
            "no",
            "zero",
            "zeros",
            "normal",
            "gauss",
            "uniform",
            "lin1d",
            "exp1d",
            "lin2d",
            "exp2d",
            "sincos",
        ],
    )


class Leddam(nn.Module):
    """
    Paper: https://arxiv.org/abs/2402.12694
    Source: https://github.com/Levi-Ackman/Leddam/blob/main/models/Leddam.py
    """

    def __init__(self, configs):
        super().__init__()
        self.leddam = DualAttention(
            configs.input_channels,
            configs.input_len,
            configs.d_model,
            configs.dropout,
            configs.positional_encoding_type,
            kernel_size=25,
            n_layers=configs.n_layers,
        )
        self.Linear_main = nn.Linear(configs.d_model, configs.output_len)
        self.Linear_res = nn.Linear(configs.d_model, configs.output_len)
        self.Linear_main.weight = nn.Parameter(
            (1 / configs.d_model) * torch.ones([configs.output_len, configs.d_model])
        )
        self.Linear_res.weight = nn.Parameter(
            (1 / configs.d_model) * torch.ones([configs.output_len, configs.d_model])
        )

    def forward(self, inp):
        res, main = self.leddam(inp)
        main_out = self.Linear_main(main.permute(0, 2, 1)).permute(0, 2, 1)
        res_out = self.Linear_res(res.permute(0, 2, 1)).permute(0, 2, 1)
        pred = main_out + res_out

        return pred
