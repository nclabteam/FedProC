import torch
from torch.optim import Optimizer


class TS_Adam(Optimizer):
    """
    TS-Adam: Adam without second-moment bias correction.

    The key heuristic: bias_correction2 = 1 (instead of 1 - β₂^t).
    Under distribution shifts this avoids over-inflating early steps,
    producing more conservative updates that generalise better across
    non-stationary time-series data.

    Reference: "Rethinking Adam for Time Series Forecasting: A Simple
    Heuristic to Improve Optimization under Distribution Shifts",
    arXiv:2603.10095.
    """

    optional = {
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "weight_decay": 0,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--beta1", type=float, default=None)
        parser.add_argument("--beta2", type=float, default=None)
        parser.add_argument("--epsilon", type=float, default=None)
        parser.add_argument("--weight_decay", type=float, default=None)

    def __init__(self, params, configs):
        defaults = dict(
            lr=configs.learning_rate,
            betas=(configs.beta1, configs.beta2),
            eps=configs.epsilon,
            weight_decay=configs.weight_decay,
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                if group["weight_decay"] != 0:
                    grad = grad.add(p.data, alpha=group["weight_decay"])

                state["step"] += 1
                t = state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Only first-moment bias correction; second moment is uncorrected
                bias_correction1 = 1 - beta1**t
                step_size = group["lr"] / bias_correction1

                denom = exp_avg_sq.sqrt().add_(group["eps"])
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
