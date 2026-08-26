import math
import warnings
from argparse import ArgumentParser, Namespace

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class HyperbolicLR(LRScheduler):
    """Apply the paper's linear-space hyperbolic learning-rate curve."""

    optional = {"upper_bound": 10, "infimum_lr": 1e-6}
    _requires_positive_infimum = False

    @staticmethod
    def args_update(parser: ArgumentParser) -> None:
        parser.add_argument("--upper_bound", type=int, default=None)
        parser.add_argument(
            "--infimum_lr",
            type=float,
            default=None,
            help="Hyperbolic learning-rate infimum.",
        )

    def __init__(
        self,
        optimizer: Optimizer,
        configs: Namespace,
        last_epoch: int = -1,
    ) -> None:
        total_epochs = configs.max_epochs
        if not isinstance(total_epochs, int) or total_epochs <= 0:
            raise ValueError("max_epochs must be a positive integer")
        self.max_iter = total_epochs - 1
        self.upper_bound = configs.upper_bound * total_epochs
        self.infimum_lr = float(configs.infimum_lr)
        if not isinstance(self.upper_bound, (int, float)) or self.upper_bound <= 0:
            raise ValueError("upper_bound must be positive")
        if self.upper_bound < self.max_iter:
            raise ValueError("upper_bound * max_epochs must cover all epochs")
        if self.infimum_lr < 0:
            raise ValueError("infimum_lr must be non-negative")
        if self._requires_positive_infimum and self.infimum_lr == 0:
            raise ValueError("infimum_lr must be positive for exponential decay")
        if any(self.infimum_lr >= group["lr"] for group in optimizer.param_groups):
            raise ValueError("infimum_lr must be below every initial learning rate")

        self._term0 = self._curve(
            iteration=0,
            max_iter=self.max_iter,
            upper_bound=self.upper_bound,
        )
        super().__init__(optimizer=optimizer, last_epoch=last_epoch)

    @staticmethod
    def _curve(iteration: int, max_iter: int, upper_bound: float) -> float:
        # Paper hyperbolic definition: h(n)=sqrt((N-n)/U * (2-(N+n)/U)).
        squared = ((max_iter - iteration) / upper_bound) * (
            2 - (max_iter + iteration) / upper_bound
        )
        return math.sqrt(max(0.0, squared))

    @staticmethod
    def _interpolate(
        base_lr: float,
        infimum_lr: float,
        curve_delta: float,
    ) -> float:
        return base_lr + (base_lr - infimum_lr) * curve_delta

    def get_lr(self) -> list[float]:
        if not self._get_lr_called_within_step:
            warnings.warn(
                message="Use get_last_lr() to read the last computed learning rate.",
                category=UserWarning,
                stacklevel=2,
            )
        iteration = min(max(self.last_epoch, 0), self.max_iter)
        curve_delta = (
            self._curve(
                iteration=iteration,
                max_iter=self.max_iter,
                upper_bound=self.upper_bound,
            )
            - self._term0
        )
        return [
            self._interpolate(
                base_lr=base_lr,
                infimum_lr=self.infimum_lr,
                curve_delta=curve_delta,
            )
            for base_lr in self.base_lrs
        ]
