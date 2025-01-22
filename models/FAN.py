import torch
import torch.nn as nn

optional = {"bias": True, "with_gate": True}


def args_update(parser):
    parser.add_argument("--bias", type=bool, default=None)
    parser.add_argument("--with_gate", type=bool, default=None)


class FAN(nn.Module):
    """
    Fourier Analysis Networks
    Paper: https://arxiv.org/abs/2410.02675
    Source: https://github.com/YihongDong/FAN/blob/main/Timeseries_Forecasting/layers/FANLayer.py
    """

    def __init__(self, configs):
        super().__init__()
        self.input_linear_p = nn.Linear(
            configs.input_len, configs.output_len // 4, bias=configs.bias
        )
        self.input_linear_g = nn.Linear(
            configs.input_len, (configs.output_len - configs.output_len // 2)
        )
        self.activation = nn.GELU()
        if configs.with_gate:
            self.gate = nn.Parameter(torch.randn(1, dtype=torch.float32))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        g = self.activation(self.input_linear_g(x))
        p = self.input_linear_p(x)

        if not hasattr(self, "gate"):
            output = torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)
        else:
            gate = torch.sigmoid(self.gate)
            output = torch.cat(
                (gate * torch.cos(p), gate * torch.sin(p), (1 - gate) * g), dim=-1
            )
        return output.permute(0, 2, 1)
