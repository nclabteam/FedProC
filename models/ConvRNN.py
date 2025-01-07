import torch
import torch.nn as nn


class ConvRNN(nn.Module):
    """
    Source: https://github.com/KurochkinAlexey/ConvRNN/blob/master/ConvRNN_SML2010.ipynb
    """

    def __init__(self, configs):
        super().__init__()
        kernel_size1 = 7
        kernel_size2 = 5
        kernel_size3 = 3
        n_channels1 = 32
        n_channels2 = 32
        n_channels3 = 32
        n_units1 = 32
        n_units2 = 32
        n_units3 = 32
        self.seq_len = configs.input_len
        self.pred_len = configs.output_len
        self.input_dim = configs.input_channels

        self.avg_pool1 = nn.AvgPool1d(2, 2)
        self.avg_pool2 = nn.AvgPool1d(4, 4)

        self.conv11 = nn.Conv1d(self.input_dim, n_channels1, kernel_size=kernel_size1)
        self.conv12 = nn.Conv1d(n_channels1, n_channels1, kernel_size=kernel_size1)

        self.conv21 = nn.Conv1d(self.input_dim, n_channels2, kernel_size=kernel_size2)
        self.conv22 = nn.Conv1d(n_channels2, n_channels2, kernel_size=kernel_size2)

        self.conv31 = nn.Conv1d(self.input_dim, n_channels3, kernel_size=kernel_size3)
        self.conv32 = nn.Conv1d(n_channels3, n_channels3, kernel_size=kernel_size3)

        self.gru1 = nn.GRU(n_channels1, n_units1, batch_first=True)
        self.gru2 = nn.GRU(n_channels2, n_units2, batch_first=True)
        self.gru3 = nn.GRU(n_channels3, n_units3, batch_first=True)

        # Linear layers
        self.linear1 = nn.Linear(
            n_units1 + n_units2 + n_units3, self.pred_len * self.input_dim
        )
        self.linear2 = nn.Linear(
            self.input_dim * self.seq_len, self.pred_len * self.input_dim
        )

        self.zp11 = nn.ConstantPad1d(((kernel_size1 - 1), 0), 0)
        self.zp12 = nn.ConstantPad1d(((kernel_size1 - 1), 0), 0)
        self.zp21 = nn.ConstantPad1d(((kernel_size2 - 1), 0), 0)
        self.zp22 = nn.ConstantPad1d(((kernel_size2 - 1), 0), 0)
        self.zp31 = nn.ConstantPad1d(((kernel_size3 - 1), 0), 0)
        self.zp32 = nn.ConstantPad1d(((kernel_size3 - 1), 0), 0)

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()

        # Permute to (batch_size, input_dim, seq_len) for 1D convolutions
        x = x.permute(0, 2, 1)

        # Flatten parameters
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
        self.gru3.flatten_parameters()

        # Line 1
        y1 = self.zp11(x)
        y1 = torch.relu(self.conv11(y1))
        y1 = self.zp12(y1)
        y1 = torch.relu(self.conv12(y1))
        y1 = y1.permute(0, 2, 1)  # Back to (batch_size, seq_len, n_channels1)
        _, h1 = self.gru1(y1)

        # Line 2
        y2 = self.avg_pool1(x)
        y2 = self.zp21(y2)
        y2 = torch.relu(self.conv21(y2))
        y2 = self.zp22(y2)
        y2 = torch.relu(self.conv22(y2))
        y2 = y2.permute(0, 2, 1)  # Back to (batch_size, seq_len/2, n_channels2)
        _, h2 = self.gru2(y2)

        # Line 3
        y3 = self.avg_pool2(x)
        y3 = self.zp31(y3)
        y3 = torch.relu(self.conv31(y3))
        y3 = self.zp32(y3)
        y3 = torch.relu(self.conv32(y3))
        y3 = y3.permute(0, 2, 1)  # Back to (batch_size, seq_len/4, n_channels3)
        _, h3 = self.gru3(y3)

        # Concatenate hidden states from all GRUs
        h = torch.cat(
            [h1[-1], h2[-1], h3[-1]], dim=1
        )  # Shape: (batch_size, n_units1+n_units2+n_units3)

        # Linear1: Based on RNN path
        out1 = self.linear1(h)  # Shape: (batch_size, pred_len * input_dim)

        # Linear2: Based on flattened input
        out2 = self.linear2(x.reshape(batch_size, -1))  # Fixed with .reshape()

        # Combine outputs
        out = out1 + out2  # Combine both paths
        out = out.view(
            batch_size, self.pred_len, self.input_dim
        )  # Reshape to (batch_size, pred_len, input_dim)

        return out
