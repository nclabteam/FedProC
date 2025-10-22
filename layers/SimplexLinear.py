import torch
import torch.nn.functional as F
from torch import nn


class SimplexLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=False):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.randn(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(output_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, input):
        # Apply softmax to the weight along the input feature dimension
        weight = torch.log(self.weight.abs() + 1)
        weight = weight / weight.sum(1, keepdim=True)
        output = F.linear(input, weight, self.bias)

        return output

    def loss(self):
        return self.weight.abs().sum()
