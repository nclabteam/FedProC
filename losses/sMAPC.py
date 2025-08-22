import torch
import torch.nn as nn


class sMAPC(nn.Module):
    """
    Symmetric Mean Absolute Percentage Change
    Paper: https://www.sciencedirect.com/science/article/abs/pii/S016920702200098X
    """

    eval_only = True

    def __init__(self):
        super(sMAPC, self).__init__()

    def forward(self, input, target):
        # y_t_n is from step 2 to h
        y_t_n = input[:, 1:]  # shape [batch, h-1]
        # y_t_n_minus_1 is previous forecast (shifted)
        y_t_n_minus_1 = input[:, :-1]  # shape [batch, h-1]

        h = input.shape[1]

        numerator = torch.abs(y_t_n - y_t_n_minus_1)
        denominator = torch.abs(y_t_n) - torch.abs(y_t_n_minus_1)

        epsilon = 1e-8
        ratio = numerator / (denominator + epsilon)

        loss = (200.0 / (h - 1)) * torch.sum(ratio, dim=-1)
        return loss.mean()


if __name__ == "__main__":
    # Example usage
    criterion = sMAPC()
    y_pred = torch.randn(32, 96, 7)  # Example predicted values
    y_true = torch.randn(32, 96, 7)  # Example true values
    loss = criterion(y_pred, y_true)
    print(loss)
