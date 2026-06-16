import torch.nn as nn


class MLP(nn.Module):
    """Plain channel-independent MLP forecaster.

    Minimal over-parameterizable nonlinear baseline. Unlike TiDE there is no
    normalization, no residual block, and no linear shortcut, so the
    memorize -> plateau -> generalize (grokking) dynamics are not damped by an
    auxiliary linear path. The same MLP is applied to every channel
    (input_len -> output_len), keeping the parameter count small and the
    capacity/data ratio easy to control.
    """

    optional = {
        "hidden_dim": 128,
        "n_layers": 2,
        "dropout": 0.0,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--hidden_dim", type=int, default=None)
        parser.add_argument("--n_layers", type=int, default=None)
        parser.add_argument("--dropout", type=float, default=None)

    def __init__(self, configs):
        super().__init__()
        in_dim = configs.input_len
        out_dim = configs.output_len
        h = configs.hidden_dim
        n = configs.n_layers
        dropout = configs.dropout

        dims = [in_dim] + [h] * n
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(a, b))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        # x: [batch, input_len, channels]
        x = x.permute(0, 2, 1)  # [batch, channels, input_len]
        x = self.net(x)  # [batch, channels, output_len]
        x = x.permute(0, 2, 1)  # [batch, output_len, channels]
        return x
