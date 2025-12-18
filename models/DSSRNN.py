import torch
import torch.nn as nn

from layers import SeriesDecompMA

optional = {
    "kernel_size": 25,
    "hidden_size": 200,
}


def args_update(parser):
    parser.add_argument("--kernel_size", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)


class DSSRNN(nn.Module):
    """
    Decomposition-Enhanced State-Space Recurrent Neural Network
    Paper: https://arxiv.org/abs/2412.00994
    Source: https://github.com/ahmad-shirazi/DSSRNN/blob/main/models/DSSRNN.py
    """

    def __init__(self, configs):
        super().__init__()
        self.decomposition = SeriesDecompMA(kernel_size=configs.kernel_size)
        self.SSRNN_Seasonal = SSRNN(
            seq_len=configs.input_len,
            pred_len=configs.output_len,
            hidden_size=configs.hidden_size,
        )
        self.Linear_Trend = nn.Linear(configs.input_len, configs.output_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decomposition(x)
        trend_init = trend_init.permute(0, 2, 1)

        seasonal_output = self.SSRNN_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        trend_output = trend_output.permute(0, 2, 1)
        x = seasonal_output + trend_output
        return x  # to [Batch, Output length, Channel]


class SSRNN(nn.Module):
    def __init__(self, seq_len, pred_len, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.rnn_cell = CustomRNNCell(input_size=seq_len, hidden_size=hidden_size)

        self.Linear = nn.Linear(seq_len, pred_len)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        # x.shape: torch.Size([16, 96, 21])
        # [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)
        h = torch.zeros(x.size(1), self.hidden_size).to(x.device)

        out = []
        for t in range(x.size(0)):
            h = self.rnn_cell(x[t, :, :].squeeze(), h)
            out.append(h.unsqueeze(1))
        out = torch.cat(out, dim=1)

        # Reshape the output to [Batch, Output length, Channel]
        out = self.fc(out)
        out = out.permute(1, 2, 0)

        return out


class CustomRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # Layers for input and hidden transformations
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, hidden):
        # Compute transformations for input and hidden state
        input_transformation = self.input_layer(x)
        hidden_transformation = self.hidden_layer(hidden)

        # Apply non-linearity (tanh is commonly used in RNNs)
        activation = torch.relu(input_transformation + hidden_transformation)

        # Sum activation output with the current state to get the next state
        out = torch.relu(activation + hidden_transformation)
        return out
