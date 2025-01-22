import torch
import torch.nn as nn

optional = {
    "hidden_size": 200,
}


def args_update(parser):
    parser.add_argument("--hidden_size", type=int, default=None)


class DSSRNN(nn.Module):
    """
    Decomposition-Enhanced State-Space Recurrent Neural Network
    Paper: https://arxiv.org/abs/2412.00994
    Source: https://github.com/ahmad-shirazi/DSSRNN/blob/main/DSSRNN-imputation/models/SSRNN.py
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.input_len
        self.pred_len = configs.output_len
        self.hidden_size = configs.hidden_size

        self.rnn_cell = CustomRNNCell(
            input_size=self.seq_len, hidden_size=self.hidden_size
        )

        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        self.fc = nn.Linear(self.hidden_size, self.pred_len)

    def forward(self, x):
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
        out = self.Linear(x).permute(0, 2, 1)
        return out


class CustomRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

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
