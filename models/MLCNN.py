import torch
import torch.nn as nn
import torch.nn.functional as F

optional = {
    "kernel_size": 3,
    "hidCNN": 10,
    "hidRNN": 25,
    "highway_window": 8,
    "collaborate_span": 2,
    "n_CNN": 5,
    "dropout": 0.3,
}


def args_update(parser):
    parser.add_argument("--kernel_size", type=int, default=None)
    parser.add_argument("--hidCNN", type=int, default=None)
    parser.add_argument("--hidRNN", type=int, default=None)
    parser.add_argument("--highway_window", type=int, default=None)
    parser.add_argument("--collaborate_span", type=int, default=None)
    parser.add_argument("--n_CNN", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)


class MLCNN(nn.Module):
    """
    Paper: https://arxiv.org/abs/1912.05122
    Source: https://github.com/smallGum/MLCNN-Multivariate-Time-Series/blob/master/models/models.py
    """

    def __init__(self, configs):
        super().__init__()
        self.idim = configs.input_channels
        self.kernel_size = configs.kernel_size
        self.hidC = configs.hidCNN
        self.hidR = configs.hidRNN
        self.hw = configs.highway_window
        self.collaborate_span = configs.collaborate_span

        self.cnn_split_num = int(configs.n_CNN / (self.collaborate_span * 2 + 1))
        self.n_CNN = self.cnn_split_num * (self.collaborate_span * 2 + 1)

        self.dropout = nn.Dropout(p=configs.dropout)
        self.convs = nn.ModuleList([])
        self.bns = nn.ModuleList([])
        for i in range(self.n_CNN):
            if i == 0:
                tmpconv = nn.Conv2d(
                    1,
                    self.hidC,
                    kernel_size=(self.kernel_size, self.idim),
                    padding=(self.kernel_size // 2, 0),
                )
            else:
                tmpconv = nn.Conv2d(
                    1,
                    self.hidC,
                    kernel_size=(self.kernel_size, self.hidC),
                    padding=(self.kernel_size // 2, 0),
                )
            self.convs.append(tmpconv)
            self.bns.append(nn.BatchNorm2d(self.hidC))

        self.shared_lstm = nn.LSTM(self.hidC, self.hidR)
        self.target_lstm = nn.LSTM(self.hidC, self.hidR)

        self.linears = nn.ModuleList([])
        self.highways = nn.ModuleList([])

        for i in range(self.collaborate_span * 2 + 1):
            self.linears.append(
                nn.Linear(self.hidR, self.idim * 96)
            )  # Now predicting the full sequence
            if self.hw > 0:
                self.highways.append(
                    nn.Linear(self.hw * (i + 1), 96)
                )  # Matching output sequence length

    def forward(self, x):
        batch_size, seq_len, _ = x.shape  # Expecting [batch, 96, 7]

        regressors = []
        currentR = torch.unsqueeze(x, 1)  # [batch, 1, 96, 7]
        for i in range(self.n_CNN):
            currentR = self.convs[i](currentR)
            currentR = self.bns[i](currentR)
            currentR = F.leaky_relu(currentR, negative_slope=0.01)
            currentR = torch.squeeze(currentR, 3)  # [batch, hidC, 96]
            if (i + 1) % self.cnn_split_num == 0:
                regressors.append(currentR)
                currentR = self.dropout(currentR)
            if i < self.n_CNN - 1:
                currentR = currentR.permute(0, 2, 1).contiguous()
                currentR = torch.unsqueeze(currentR, 1)

        shared_lstm_results = []
        target_R = None
        target_h = None
        target_c = None

        self.shared_lstm.flatten_parameters()
        for i in range(self.collaborate_span * 2 + 1):
            cur_R = regressors[i].permute(2, 0, 1).contiguous()  # [96, batch, hidC]
            _, (cur_result, cur_state) = self.shared_lstm(cur_R)
            if i == self.collaborate_span:
                target_R = cur_R
                target_h = cur_result
                target_c = cur_state
            cur_result = self.dropout(torch.squeeze(cur_result, 0))
            shared_lstm_results.append(cur_result)

        self.target_lstm.flatten_parameters()
        _, (target_result, _) = self.target_lstm(target_R, (target_h, target_c))
        target_result = self.dropout(torch.squeeze(target_result, 0))

        res = None
        for i in range(self.collaborate_span * 2 + 1):
            if i == self.collaborate_span:
                cur_res = self.linears[i](
                    target_result
                )  # [batch, hidR] → [batch, 96 * idim]
            else:
                cur_res = self.linears[i](shared_lstm_results[i])

            cur_res = cur_res.view(
                batch_size, 96, self.idim
            )  # Reshape to [batch, 96, 7]
            cur_res = torch.unsqueeze(
                cur_res, 1
            )  # Add collaboration dimension: [batch, 1, 96, 7]

            res = (
                cur_res if res is None else torch.cat((res, cur_res), 1)
            )  # [batch, 5, 96, 7]

        res = res.permute(0, 2, 1, 3).contiguous()  # [batch, 96, 5, 7]
        res = torch.mean(
            res, dim=2
        )  # Aggregate across collaboration span → [batch, 96, 7]

        # Highway Connection
        if self.hw > 0:
            highway = None
            for i in range(self.collaborate_span * 2 + 1):
                z = x[:, -(self.hw * (i + 1)) :, :]  # Take last `hw * (i+1)` timesteps
                z = (
                    z.permute(0, 2, 1).contiguous().view(batch_size * self.idim, -1)
                )  # Flatten time
                z = self.highways[i](z)  # [batch*idim, 96]
                z = z.view(batch_size, self.idim, 96).permute(
                    0, 2, 1
                )  # [batch, 96, idim]

                if highway is None:
                    highway = torch.unsqueeze(z, 1)  # [batch, 1, 96, idim]
                else:
                    highway = torch.cat(
                        (highway, torch.unsqueeze(z, 1)), 1
                    )  # [batch, 5, 96, idim]

            highway = torch.mean(
                highway, dim=1
            )  # Reduce across collaboration span → [batch, 96, idim]
            res = res + highway  # Ensure highway size matches output

        return res  # Final output: [batch, 96, 7]
