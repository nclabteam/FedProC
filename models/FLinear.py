import torch.nn as nn

from layers import FAN

optional = {
    "freq_topk": 20,
    "rfft": True,
}


def args_update(parser):
    parser.add_argument("--freq_topk", type=int, default=None)
    parser.add_argument("--rfft", type=bool, default=None)


class FLinear(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.Linear = nn.Linear(self.input_len, self.output_len)
        self.rev = FAN(
            seq_len=configs.seq_len,
            pred_len=configs.pred_len,
            enc_in=configs.input_channels,
            freq_topk=self.freq_topk,
            rfft=self.rfft,
        )

    def forward(self, x):
        # x: [B, seq_len, in_channels]
        x = self.rev(x, "norm")
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.rev(x, "denorm")
        return x  # [B, pred_len, out_channels]
