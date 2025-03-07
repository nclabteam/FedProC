import torch
import torch.nn as nn

optional = {"d_model": 128, "period_len": 24, "sparse_type": "linear"}


def args_update(parser):
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--period_len", type=int, default=None)
    parser.add_argument(
        "--sparse_type", type=str, default=None, choices=["linear", "mlp"]
    )


class SparseTSF(nn.Module):
    """
    Paper: https://arxiv.org/abs/2405.00946
    Source: https://github.com/lss-1138/SparseTSF/blob/main/models/SparseTSF.py
    """

    def __init__(self, configs):
        super().__init__()

        # get parameters
        self.seq_len = configs.input_len
        self.pred_len = configs.output_len
        self.enc_in = configs.input_channels
        self.period_len = configs.period_len
        self.d_model = configs.d_model
        self.sparse_type = configs.sparse_type

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1,
            padding=self.period_len // 2,
            padding_mode="zeros",
            bias=False,
        )

        if self.sparse_type == "linear":
            self.net = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)
        elif self.sparse_type == "mlp":
            self.net = nn.Sequential(
                nn.Linear(self.seg_num_x, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.seg_num_y),
            )

    def forward(self, x):
        batch_size = x.shape[0]
        # normalization and permute     b,s,c -> b,c,s
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)

        # 1D convolution aggregation
        x = (
            self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(
                -1, self.enc_in, self.seq_len
            )
            + x
        )

        # downsampling: b,c,s -> bc,n,w -> bc,w,n
        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)

        # sparse forecasting
        y = self.net(x)

        # upsampling: bc,w,m -> bc,m,w -> b,c,s
        y = y.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)

        # permute and denorm
        y = y.permute(0, 2, 1) + seq_mean

        return y
