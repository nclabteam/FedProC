import torch
from torch import nn

from .DLinear import DLinear


class MTSD(DLinear):

    optional = {
        **DLinear.optional,
        "d_model": 512,
    }

    @classmethod
    def args_update(cls, parser):
        DLinear.args_update(parser)
        parser.add_argument("--d_model", type=int, default=None)

    def __init__(self, configs):
        super().__init__(configs)
        self.Linear_Trend = nn.Sequential(
            nn.Linear(self.seq_len, configs.d_model),
            nn.GELU(),
            nn.Linear(configs.d_model, self.pred_len),
        )
        self.Linear_Trend.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
        )
