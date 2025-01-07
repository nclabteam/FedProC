import torch
from torch import nn


class RMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, input, target):
        return torch.sqrt(self.mse(input=input, target=target))
