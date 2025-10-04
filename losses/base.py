import numpy as np
import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = torch.tensor(torch.finfo(torch.float64).eps)

    def forward(self, input, target):
        raise NotImplementedError

    def divide_no_nan(self, a, b):
        """
        a/b where the resulted NaN or Inf are replaced by 0.
        """
        result = a / b
        result[result != result] = 0.0
        result[result == np.inf] = 0.0
        return result

    def _percentage_error(self, input, target):
        return self.divide_no_nan(a=target - input, b=target) * 100

    def _log_error(self, x, y):
        return self.divide_no_nan(a=x, b=y)
