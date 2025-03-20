from layers import RevIN

from .DLinear import DLinear, args_update, optional


class RDLinear(DLinear):
    """
    Paper: https://ieeexplore.ieee.org/document/10650961
    """

    def __init__(self, configs):
        super().__init__(configs=configs)
        self.rev = RevIN(num_features=configs.input_channels)

    def forward(self, x):
        seasonal_init, trend_init = self.decompsition(x)

        trend_init = self.rev(trend_init, mode="norm")

        trend_init = trend_init.permute(0, 2, 1)
        seasonal_init = seasonal_init.permute(0, 2, 1)

        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        trend_output = trend_output.permute(0, 2, 1)
        trend_output = self.rev(trend_output, mode="denorm")
        trend_output = trend_output.permute(0, 2, 1)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # [B, L, D]
