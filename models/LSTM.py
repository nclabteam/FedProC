import torch
import torch.nn as nn

optional = {
    "hidden_size": 128,
    "num_layers": 2,
}


def args_update(parser):
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)


class LSTM(nn.Module):
    def __init__(self, configs):
        super(LSTM, self).__init__()
        self.hidden_size = configs.hidden_size
        self.num_layers = configs.num_layers
        self.pred_len = configs.input_len
        self.enc_in = configs.input_channels

        self.cells = nn.ModuleList(
            [
                LSTMCell(self.enc_in, self.hidden_size)
                if i == 0
                else LSTMCell(self.hidden_size, self.hidden_size)
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
        c = [
            torch.zeros(batch_size, self.hidden_size).to(x.device)
            for _ in range(self.num_layers)
        ]

        # Process the input sequence through LSTM
        for t in range(seq_len):
            x_t = x[:, t, :]
            for i, cell in enumerate(self.cells):
                h[i], (h[i], c[i]) = cell(x_t, (h[i], c[i]))
                x_t = h[i]

        # Use the final hidden state to predict all time steps at once
        last_hidden_state = h[-1]  # Shape: (batch_size, hidden_size)
        pred = self.fc_pred(last_hidden_state)  # Shape: (batch_size, pred_len * enc_in)
        pred = pred.view(
            batch_size, self.pred_len, self.enc_in
        )  # Reshape to desired output

        return pred  # Shape: (batch_size, pred_len, enc_in)


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        # Input gate components
        self.W_ii = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        # Forget gate components
        self.W_if = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        # Cell gate components
        self.W_ig = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_g = nn.Parameter(torch.Tensor(hidden_size))

        # Output gate components
        self.W_io = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        self.init_weights()

    def init_weights(self):
        for param in self.parameters():
            nn.init.uniform_(param, -0.1, 0.1)

    def forward(self, x, hidden):
        h_prev, c_prev = hidden

        i_t = torch.sigmoid(x @ self.W_ii.T + h_prev @ self.W_hi.T + self.b_i)
        f_t = torch.sigmoid(x @ self.W_if.T + h_prev @ self.W_hf.T + self.b_f)
        g_t = torch.tanh(x @ self.W_ig.T + h_prev @ self.W_hg.T + self.b_g)
        o_t = torch.sigmoid(x @ self.W_io.T + h_prev @ self.W_ho.T + self.b_o)

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, (h_t, c_t)
