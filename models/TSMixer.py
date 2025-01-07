import torch.nn as nn

optional = {
    "d_model": 128,
    "num_layers": 5,
    "dropout": 0.2,
}


def args_update(parser):
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)


class TSMixer(nn.Module):
    """
    Source: https://github.com/decisionintelligence/TFB/blob/master/ts_benchmark/baselines/time_series_library/models/TSMixer.py
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.input_len
        self.pred_len = configs.output_len
        self.num_layers = configs.num_layers

        self.model = nn.ModuleList([ResBlock(configs) for _ in range(self.num_layers)])
        self.projection = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [B, L, D]
        for i in range(self.num_layers):
            x = self.model[i](x)
        return self.projection(x.transpose(1, 2)).transpose(1, 2)  # [B, L, D]


class ResBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.temporal_mlp = nn.Sequential(
            nn.Linear(configs.input_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.input_len),
            nn.Dropout(configs.dropout),
        )

        self.feature_mlp = nn.Sequential(
            nn.Linear(configs.input_channels, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.input_channels),
            nn.Dropout(configs.dropout),
        )

    def forward(self, x):
        # x: [B, L, D]
        x = x + self.temporal_mlp(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.feature_mlp(x)
        return x
