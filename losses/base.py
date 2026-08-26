import torch
from torch import Tensor, nn


class Loss(nn.Module):
    """Provide shared tensor formulas for forecast losses."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def divide_no_nan(a: Tensor, b: Tensor) -> Tensor:
        """Divide tensors while replacing undefined results with zero."""
        nonzero = b != 0
        quotient = a / torch.where(nonzero, b, torch.ones_like(b))
        quotient = torch.nan_to_num(quotient, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.where(nonzero, quotient, torch.zeros_like(quotient))

    @staticmethod
    def _percentage_error(input: Tensor, target: Tensor) -> Tensor:
        return Loss.divide_no_nan(a=target - input, b=target) * 100

    @staticmethod
    def _symmetric_absolute_percentage_error(
        input: Tensor,
        target: Tensor,
    ) -> Tensor:
        return Loss.divide_no_nan(
            a=200 * torch.abs(target - input),
            b=torch.abs(target) + torch.abs(input),
        )

    @staticmethod
    def _modified_symmetric_absolute_percentage_error(
        input: Tensor,
        target: Tensor,
    ) -> Tensor:
        # Forecast Evaluation msMAPE definition: epsilon=0.1, floor=0.5+epsilon.
        denominator = (torch.abs(target) + torch.abs(input) + 0.1).clamp_min(0.6)
        return 200 * torch.abs(target - input) / denominator

    @staticmethod
    def _log_error(input: Tensor, target: Tensor) -> Tensor:
        # MALE/RMSLE definition: log(forecast) - log(observation).
        return torch.log(input) - torch.log(target)
