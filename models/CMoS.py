import torch
import torch.nn as nn
import torch.nn.functional as F


class CMoS(nn.Module):
    """
    CMoS: Rethinking Time Series Prediction Through the Lens of
    Chunk-wise Spatial Correlations (ICML 2025).

    Super-lightweight model using chunk-wise spatial correlation mapping
    with correlation mixing via foundation matrices.
    """

    optional = {
        "seg_size": 8,
        "num_map": 4,
        "kernel_size": 8,
        "conv_stride": 4,
        "use_pi": False,
        "period": 24,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--seg_size", type=int, default=None)
        parser.add_argument("--num_map", type=int, default=None)
        parser.add_argument("--kernel_size", type=int, default=None)
        parser.add_argument("--conv_stride", type=int, default=None)
        parser.add_argument("--use_pi", type=int, default=None, choices=[0, 1])
        parser.add_argument("--period", type=int, default=None)

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.input_len
        self.pred_len = configs.output_len
        self.c = configs.input_channels

        self.seg_size = configs.seg_size
        self.num_map = configs.num_map
        self.kernel_size = configs.kernel_size
        self.conv_stride = configs.conv_stride

        # Foundation correlation matrices
        in_chunks = self.seq_len // self.seg_size
        out_chunks = self.pred_len // self.seg_size
        num_mappings = self.num_map + 1 if configs.use_pi else self.num_map
        self.mappings = nn.ModuleList(
            [nn.Linear(in_chunks, out_chunks) for _ in range(num_mappings)]
        )

        # Periodicity injection
        if configs.use_pi:
            period = configs.period
            stride = period // self.seg_size
            new_weights = torch.zeros(out_chunks, in_chunks)
            for i in range(out_chunks):
                for j in range(in_chunks - stride, 0, -stride):
                    if j + i < in_chunks:
                        new_weights[i, j + i] = period / self.seq_len
            self.mappings[0].weight.data = new_weights
            self.mappings[0].bias.data.zero_()

        # Per-channel depthwise conv for gating
        self.conv_dim = (self.seq_len - self.kernel_size) // self.conv_stride + 1
        self.ds_convs = nn.ModuleList(
            [
                nn.Conv1d(1, 1, self.kernel_size, stride=self.conv_stride)
                for _ in range(self.c)
            ]
        )

        # Gating network
        self.gates = nn.Linear(self.conv_dim, self.num_map)

    def forward(self, x, **kwargs):
        # x: [B, seq_len, c]
        x = x.transpose(1, 2)  # [B, c, seq_len]

        # RevIN normalization
        means = x.mean(2, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=2, keepdim=True, unbiased=False) + 1e-10)
        x = x / stdev

        # Per-channel conv for gating
        conv_out = torch.cat(
            [self.ds_convs[i](x[:, i : i + 1, :]) for i in range(self.c)], dim=1
        )
        gates_out = F.softmax(
            self.gates(conv_out.squeeze(1)), dim=-1
        )  # [B, c, num_map]

        # Chunk-wise mapping
        bs = x.size(0)
        x_ = x.reshape(bs, self.c, -1, self.seg_size).transpose(2, 3)
        x_out = torch.stack(
            [
                self.mappings[i](x_).transpose(2, 3).flatten(start_dim=2)
                for i in range(self.num_map)
            ],
            dim=2,
        )  # [B, c, num_map, pred_len]

        # Correlation mixing
        x = torch.einsum("bcns,bcn->bcs", x_out, gates_out)  # [B, c, pred_len]

        # De-normalization
        x = x * stdev + means
        return x.transpose(1, 2)  # [B, pred_len, c]
