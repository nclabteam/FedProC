import math
import warnings

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

from .HyperbolicLR import args_update, optional


class ExpHyperbolicLR(_LRScheduler):
    """
    Paper: https://arxiv.org/abs/2407.15200
    Source: https://github.com/Axect/HyperbolicLR/blob/main/hyperbolic_lr.py
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
        # Allow infimum_lr == 0 for exponential decay to zero
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
        # Add epsilon for division stability
        self._term0 = max(self._term0, 1e-12)

        # 4. Call parent class __init__ AFTER setting scheduler params
        super().__init__(optimizer=optimizer, last_epoch=last_epoch)

        # 5. Perform validation dependent on base_lrs (set by super().__init__)
        for base_lr in self.base_lrs:
            # Allow infimum_lr == base_lr only if infimum_lr is 0 (constant zero LR)
            if self.infimum_lr > base_lr:
                raise ValueError(
                    f"infimum_lr ({self.infimum_lr}) must be less than or equal to all base_lrs ({self.base_lrs})."
                )
            if self.infimum_lr == base_lr and self.infimum_lr > 0:
                warnings.warn(
                    f"infimum_lr ({self.infimum_lr}) is equal to base_lr ({base_lr}). LR will remain constant."
                )

    def get_lr(self):
        """Compute learning rate based on the current epoch (self.last_epoch)."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        current_iter = min(self.last_epoch, self.max_iter)
        N = float(self.max_iter)
        U = float(self.upper_bound)
        x = float(current_iter)

        # Calculate scale_factor(x) = term(x) / term(0)
        if current_iter >= self.max_iter:
            # At or beyond max_iter, scale factor is 0
            scale_factor = 0.0
        else:
            termx_squared_arg = ((N - x) / U) * (2.0 - (N + x) / U)
            # Handle potential floating point issues near x=N
            termx_squared_arg = max(0.0, termx_squared_arg)
            termx = math.sqrt(termx_squared_arg)
            scale_factor = termx / self._term0  # Ranges from 1 down to 0

        # Calculate the exponent for the decay: 1 - scale_factor(x)
        # This exponent ranges from 0 (at x=0) up to 1 (at x=max_iter)
        decay_exponent = 1.0 - scale_factor

        new_lrs = []
        for base_lr in self.base_lrs:
            if base_lr <= 0:  # Handle cases where base_lr is already zero or negative
                new_lrs.append(base_lr)
                continue
            # Prevent division by zero if base_lr is extremely small but positive
            if base_lr <= self.infimum_lr:
                # If target is already reached or passed, stay at infimum_lr
                # or if base_lr is zero. Handles infimum_lr == base_lr case too.
                new_lr = self.infimum_lr if x >= self.max_iter else base_lr
                new_lrs.append(max(0.0, self.infimum_lr, new_lr))  # Ensure non-negative
                continue

            # Calculate lr_ratio = target_lr / start_lr
            lr_ratio = self.infimum_lr / base_lr
            # Ensure ratio is non-negative (should be by validation, but safe check)
            lr_ratio = max(0.0, lr_ratio)

            # Calculate new LR: base_lr * ratio ^ exponent
            # Use max(0.0, ...) exponent for safety with 0**neg -> error
            try:
                # Handle 0**0 case (when infimum_lr=0 and x=0 -> decay_exponent=0) -> should be 1
                if lr_ratio == 0.0 and decay_exponent == 0.0:
                    multiplier = 1.0
                else:
                    multiplier = lr_ratio**decay_exponent

                new_lr = base_lr * multiplier
            except ValueError:  # Catches potential issues like negative base in power
                warnings.warn(
                    f"ValueError during LR calculation. ratio={lr_ratio}, exponent={decay_exponent}. Setting LR to infimum."
                )
                new_lr = self.infimum_lr

            # Ensure LR doesn't go below infimum due to float precision
            # And ensure it doesn't go *above* base_lr if decay_exponent is somehow negative (shouldn't happen)
            new_lr = max(self.infimum_lr, new_lr)
            # new_lr = min(base_lr, new_lr) # Optional: safety check

            new_lrs.append(new_lr)

        return new_lrs
