import math
import warnings

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

# --- Default configuration values ---
optional = {
    "upper_bound": 10,
    "infimum_lr": 1e-6,  # Default: Minimum learning rate
}


# --- Function to add arguments to argparse ---
def args_update(parser):
    """Adds HyperbolicLR specific arguments to the parser."""
    # Use default=None, so we know if the user provided the argument or not
    parser.add_argument("--upper_bound", type=int, default=None)
    parser.add_argument(
        "--infimum_lr",
        type=float,
        default=None,
        help=f"HyperbolicLR: Minimum learning rate",
    )


class HyperbolicLR(_LRScheduler):
    """
    Sets the learning rate according to a hyperbolic schedule, integrated with
    argparse configuration.

    The learning rate at iteration `x` is calculated as:
    LR(x) = infimum_lr + (base_lr - infimum_lr) * scale_factor(x)
    where scale_factor(x) decreases hyperbolically from 1 (at x=0) to 0 (at x=max_iter).

    scale_factor(x) = sqrt(((N - x)/U) * (2 - (N + x)/U)) / sqrt((N/U) * (2 - N/U))
    for 0 <= x <= N, where N = max_iter, U = upper_bound.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        configs (Namespace): Configuration object, typically from argparse.
                             Expected attributes: upper_bound, max_iter, infimum_lr.
                             If an attribute is None or missing, the default from 'optional' is used.
                             Optionally uses 'verbose_scheduler' flag from configs.
        last_epoch (int, optional): The index of the last epoch. Default: -1.
    """

    def __init__(self, optimizer: Optimizer, configs, last_epoch: int = -1):
        # 1. Resolve parameter values from configs or defaults
        self.upper_bound = configs.upper_bound * configs.max_epochs
        self.max_iter = configs.max_epochs
        self.infimum_lr = configs.infimum_lr

        # 2. Perform parameter validation (basic types and relations)
        if not isinstance(self.upper_bound, int) or self.upper_bound <= 0:
            raise ValueError("upper_bound must be a positive integer.")
        if not isinstance(self.max_iter, int) or self.max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")
        if self.upper_bound <= self.max_iter:
            raise ValueError(
                f"upper_bound ({self.upper_bound}) must be strictly greater than max_iter ({self.max_iter})."
            )
        if not isinstance(self.infimum_lr, (float, int)) or self.infimum_lr < 0:
            raise ValueError("infimum_lr must be a non-negative number.")

        # 3. Calculate the constant term f(0) needed for normalization
        N = float(self.max_iter)
        U = float(self.upper_bound)
        term0_squared_arg = (N / U) * (2.0 - N / U)
        if term0_squared_arg <= 0:
            raise ValueError(
                f"Invalid parameters: N={N}, U={U} result in non-positive sqrt argument for initial term ({term0_squared_arg}). Check if U > N."
            )
        self._term0 = math.sqrt(term0_squared_arg)
        # Add epsilon for division stability, although mathematically _term0 should be > 0
        self._term0 = max(self._term0, 1e-12)

        # 4. Call parent class __init__ AFTER setting scheduler params
        # This initializes self.base_lrs and handles verbose flag
        super().__init__(optimizer=optimizer, last_epoch=last_epoch)

        # 5. Perform validation dependent on base_lrs (set by super().__init__)
        for base_lr in self.base_lrs:
            if self.infimum_lr >= base_lr:
                raise ValueError(
                    f"infimum_lr ({self.infimum_lr}) must be less than all base_lrs ({self.base_lrs})."
                )

    def get_lr(self):
        """Compute learning rate based on the current epoch (self.last_epoch)."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        # Use self.last_epoch which is the current step/iteration count
        # Clamp epoch to the range [0, max_iter] for calculation
        current_iter = min(self.last_epoch, self.max_iter)

        # Calculate scale factor: term(x) / term(0)
        # term(x) = sqrt(((N - x)/U) * (2 - (N + x)/U))
        N = float(self.max_iter)
        U = float(self.upper_bound)
        x = float(current_iter)

        if current_iter >= self.max_iter:
            # At or beyond max_iter, scale factor is 0
            scale_factor = 0.0
        else:
            termx_squared_arg = ((N - x) / U) * (2.0 - (N + x) / U)
            # Handle potential floating point issues near x=N
            termx_squared_arg = max(0.0, termx_squared_arg)
            termx = math.sqrt(termx_squared_arg)
            scale_factor = termx / self._term0

        # Calculate new LR for each parameter group
        new_lrs = []
        for base_lr in self.base_lrs:
            delta_lr = base_lr - self.infimum_lr
            new_lr = self.infimum_lr + delta_lr * scale_factor
            # Ensure LR doesn't go below infimum due to potential float precision issues
            new_lrs.append(max(self.infimum_lr, new_lr))

        return new_lrs
