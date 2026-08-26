from argparse import ArgumentParser, Namespace
from typing import Any

import torch
import torch.nn as nn

from layers import DualAttention


class Leddam(nn.Module):
    """Forecast with learnable decomposition and dual attention."""

    optional = {
        "d_model": 512,
        "n_layers": 1,
        "dropout": 0,
        "positional_encoding_type": "sincos",
    }

    @classmethod
    def args_update(cls, parser: ArgumentParser) -> None:
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

    def __init__(self, configs: Namespace) -> None:
        super().__init__()
        self.leddam = DualAttention(
            enc_in=configs.input_channels,
            seq_len=configs.input_len,
            d_model=configs.d_model,
            dropout=configs.dropout,
            pe_type=configs.positional_encoding_type,
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

    def forward(self, inp: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        residual, main = self.leddam(inp=inp)
        main_output = self.Linear_main(input=main.permute(0, 2, 1)).permute(0, 2, 1)
        residual_output = self.Linear_res(input=residual.permute(0, 2, 1)).permute(
            0, 2, 1
        )
        return main_output + residual_output
