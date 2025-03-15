import torch
from torch import nn

from .DLinear import DLinear
from .DLinear import args_update as DLinear_args_update
from .DLinear import optional as DLinear_optional

optional = {
    **DLinear_optional,
    "d_model": 512,
}


def args_update(parser):
    DLinear_args_update(parser)
    parser.add_argument("--d_model", type=int, default=None)


class MTSD(DLinear):
    """
    Paper: https://arxiv.org/abs/2302.04501
    Source: https://github.com/plumprc/MTS-Mixers/blob/main/models/MTSD.py
    """

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
