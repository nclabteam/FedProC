import math

from .HyperbolicLR import HyperbolicLR


class ExpHyperbolicLR(HyperbolicLR):
    """Apply the paper's exponential-space hyperbolic learning-rate curve."""

    _requires_positive_infimum = True

    @staticmethod
    def _interpolate(
        base_lr: float,
        infimum_lr: float,
        curve_delta: float,
    ) -> float:
        return base_lr * math.exp(math.log(base_lr / infimum_lr) * curve_delta)
