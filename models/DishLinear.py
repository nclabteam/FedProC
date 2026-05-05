import torch.nn as nn

from layers import DishTS


class DishLinear(nn.Module):

    optional = {
        "dishts": "uniform",
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument(
            "--dishts", type=str, default=None, choices=["uniform", "avg", "standard"]
        )

    def __init__(self, configs):
        super().__init__()
        self.Linear = nn.Linear(configs.input_len, configs.output_len)
        self.dish = DishTS(
            num_features=configs.input_channels,
            seq_len=configs.input_len,
            dish_init=configs.dishts,
        )

    def forward(self, x, **kwargs):
        # x: [B, seq_len, in_channels]
        x = self.dish(x, "norm")
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.dish(x, "denorm")
        return x  # [B, pred_len, out_channels]
