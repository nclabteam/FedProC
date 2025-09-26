import torch
import torch.nn as nn

optional = {
    "hidden_size": 128,
    "num_layers": 2,
}


def args_update(parser):
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)


class GRU(nn.Module):
    def __init__(self, configs):
        super(GRU, self).__init__()
        self.hidden_size = configs.hidden_size
        self.num_layers = configs.num_layers
        self.pred_len = configs.output_len
        self.enc_in = configs.input_channels

        self.cells = nn.ModuleList(
            [
                (
                    GRUCell(self.enc_in, self.hidden_size)
                    if i == 0
                    else GRUCell(self.hidden_size, self.hidden_size)
                )
                for i in range(self.num_layers)
            ]
        )

        # New linear layer to project the final hidden state to (pred_len * enc_in)
        self.fc_pred = nn.Linear(self.hidden_size, self.pred_len * self.enc_in)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = [
            torch.zeros(batch_size, self.hidden_size).to(x.device)
            for _ in range(self.num_layers)
        ]

        # Process the input sequence through GRU
        for t in range(seq_len):
            x_t = x[:, t, :]
            for i, cell in enumerate(self.cells):
                h[i] = cell(x_t, h[i])
                x_t = h[i]

        # Use the final hidden state to predict all time steps at once
        last_hidden_state = h[-1]  # Shape: (batch_size, hidden_size)
        pred = self.fc_pred(last_hidden_state)  # Shape: (batch_size, pred_len * enc_in)
        # Reshape to desired output
        pred = pred.view(batch_size, self.pred_len, self.enc_in)
        return pred  # Shape: (batch_size, pred_len, enc_in)


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        # Reset gate components
        self.W_ir = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hr = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_r = nn.Parameter(torch.Tensor(hidden_size))

        # Update gate components
        self.W_iz = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hz = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_z = nn.Parameter(torch.Tensor(hidden_size))

        # New gate components
        self.W_in = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hn = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_n = nn.Parameter(torch.Tensor(hidden_size))

        self.init_weights()

    def init_weights(self):
        for param in self.parameters():
            nn.init.uniform_(param, -0.1, 0.1)

    def forward(self, x, h_prev):
        # Reset gate
        r_t = torch.sigmoid(x @ self.W_ir.T + h_prev @ self.W_hr.T + self.b_r)

        # Update gate
        z_t = torch.sigmoid(x @ self.W_iz.T + h_prev @ self.W_hz.T + self.b_z)

        # Candidate activation
        n_t = torch.tanh(x @ self.W_in.T + (r_t * h_prev) @ self.W_hn.T + self.b_n)

        # Final hidden state
        h_t = (1 - z_t) * n_t + z_t * h_prev

        return h_t
