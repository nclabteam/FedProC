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

    def _symmetric_absolute_percentage_error(self, input, target):
        return torch.div(
            200 * torch.abs(target - input),
            (torch.abs(target) + torch.abs(input)),
        )

    def _modified_symmetric_absolute_percentage_error(self, input, target):
        return torch.div(
            200 * torch.abs(target - input),
            torch.max(
                torch.abs(target) + torch.abs(input) + self.eps,
                0.5 + self.eps,
            ),
        )

    def _log_error(self, x, y):
        return self.divide_no_nan(a=torch.log(x), b=torch.log(y))
