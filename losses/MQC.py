import torch
import torch.nn as nn


class MQC(nn.Module):
    """
    Multi-Quantile Loss Change
    Paper: https://arxiv.org/abs/2506.05776
    """

    def __init__(self, quantiles=None):
        """
        quantiles: list or tensor of quantile values.
                   If None, defaults to median + 6 central prediction intervals:
                   0.005, 0.025, 0.05, 0.1, 0.15, 0.2, 0.5, 0.8, 0.85, 0.9, 0.95, 0.975, 0.995
        """
        super(MQC, self).__init__()
        if quantiles is None:
            quantiles = [
                0.005,
                0.025,
                0.05,
                0.1,
                0.15,
                0.2,
                0.5,
                0.8,
                0.85,
                0.9,
                0.95,
                0.975,
                0.995,
            ]
        self.register_buffer("quantiles", torch.tensor(quantiles).view(1, 1, -1))

    def forward(self, input, target):
        batch_size, h, features = input.shape

        # Adjust quantiles to match the number of features
        if features != self.quantiles.size(-1):
            # If features don't match quantiles, use a subset or repeat
            if features <= self.quantiles.size(-1):
                # Use the first 'features' quantiles
                q = self.quantiles[:, :, :features].to(input.device)
            else:
                # Repeat quantiles to match features
                repeat_factor = (
                    features + self.quantiles.size(-1) - 1
                ) // self.quantiles.size(-1)
                q = self.quantiles.repeat(1, 1, repeat_factor)[:, :, :features].to(
                    input.device
                )
        else:
            q = self.quantiles.to(input.device)

        y_t_n = input[:, 1:, :]  # [batch, h-1, features]
        y_t_n_m1 = input[:, :-1, :]  # [batch, h-1, features]

        cond_ge = (y_t_n_m1 >= y_t_n).float()
        cond_lt = 1.0 - cond_ge

        term1 = q * (y_t_n_m1 - y_t_n) * cond_ge
        term2 = (1 - q) * (y_t_n - y_t_n_m1) * cond_lt
        qc = (term1 + term2).mean(dim=1)  # mean over horizon

        mqc = qc.mean(dim=-1)  # mean over features
        return mqc.mean()  # mean over batch


if __name__ == "__main__":
    # Example usage
    criterion = MQC()
    y_pred = torch.randn(32, 96, 13)  # Example predicted values
    y_true = torch.randn(32, 96, 13)  # Example true values
    loss = criterion(y_pred, y_true)
    print(loss)
