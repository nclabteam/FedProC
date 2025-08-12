import torch
import torch.nn as nn


class RMSSE(nn.Module):
    """
    Root Mean Squared Scaled Error

    RMSSE = sqrt(MSE / naive_MSE)

    Where:
    - MSE is the mean squared error of predictions
    - naive_MSE is the mean squared error of naive (seasonal) forecast

    The naive forecast is typically the value from the same season in the previous period.
    For simplicity, we use the previous timestep as the naive forecast.
    """

    def __init__(self, seasonal_period=1):
        """
        Args:
            seasonal_period (int): Period for seasonal naive forecast.
                                 1 means using previous timestep,
                                 24 for hourly data with daily seasonality, etc.
        """
        super(RMSSE, self).__init__()
        self.seasonal_period = seasonal_period

    def forward(self, input, target):
        """
        Args:
            input: Predicted values [batch_size, seq_len, features]
            target: True values [batch_size, seq_len, features]

        Returns:
            RMSSE value
        """
        # Calculate MSE between predictions and targets
        mse_pred = torch.mean((input - target) ** 2)

        # Calculate naive forecast MSE
        # Naive forecast: use value from seasonal_period steps ago
        if target.size(1) <= self.seasonal_period:
            # If sequence is too short, use previous timestep
            naive_forecast = target[:, :-1, :]
            naive_target = target[:, 1:, :]
        else:
            # Use seasonal naive forecast
            naive_forecast = target[:, : -self.seasonal_period, :]
            naive_target = target[:, self.seasonal_period :, :]

        # Calculate MSE for naive forecast
        naive_mse = torch.mean((naive_forecast - naive_target) ** 2)

        # Avoid division by zero
        naive_mse = torch.clamp(naive_mse, min=1e-8)

        # Calculate RMSSE
        rmsse = torch.sqrt(mse_pred / naive_mse)

        return rmsse


if __name__ == "__main__":
    # Example usage
    batch_size, seq_len, features = 32, 96, 7

    # Create sample data
    y_pred = torch.randn(batch_size, seq_len, features)
    y_true = torch.randn(batch_size, seq_len, features)

    # Test basic RMSSE
    criterion = RMSSE(seasonal_period=1)
    loss = criterion(y_pred, y_true)
    print(f"RMSSE Loss: {loss.item():.4f}")
