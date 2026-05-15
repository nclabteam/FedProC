import torch
import torch.nn as nn
from torch.optim import Optimizer


class TS_Yogi(Optimizer):
    """
    TS-Yogi: Yogi optimizer without second-moment bias correction.

    Yogi uses an additive second-moment update rule that controls variance
    more tightly than Adam. TS-Yogi applies the same heuristic as TS-Adam
    (bias_correction2 = 1) to Yogi.

    Reference: "Rethinking Adam for Time Series Forecasting: A Simple
    Heuristic to Improve Optimization under Distribution Shifts",
    arXiv:2603.10095.
    """

    optional = {
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-3,
        "weight_decay": 0,
        "initial_accumulator": 1e-6,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--beta1", type=float, default=None)
        parser.add_argument("--beta2", type=float, default=None)
        parser.add_argument("--epsilon", type=float, default=None)
        parser.add_argument("--weight_decay", type=float, default=None)
        parser.add_argument("--initial_accumulator", type=float, default=None)

    def __init__(self, params, configs):
        defaults = dict(
            lr=configs.learning_rate,
            betas=(configs.beta1, configs.beta2),
            eps=configs.epsilon,
            weight_decay=configs.weight_decay,
            initial_accumulator=configs.initial_accumulator,
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
                    state["exp_avg"] = nn.init.constant_(
                        torch.empty_like(p.data, memory_format=torch.preserve_format),
                        group["initial_accumulator"],
                    )
                    state["exp_avg_sq"] = nn.init.constant_(
                        torch.empty_like(p.data, memory_format=torch.preserve_format),
                        group["initial_accumulator"],
                    )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]

                if group["weight_decay"] != 0:
                    grad = grad.add(p.data, alpha=group["weight_decay"])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                grad_squared = grad.mul(grad)
                # Yogi additive update: sign controls direction of correction
                exp_avg_sq.addcmul_(
                    torch.sign(exp_avg_sq - grad_squared),
                    grad_squared,
                    value=-(1 - beta2),
                )

                denom = exp_avg_sq.sqrt().add_(group["eps"])
                step_size = group["lr"] / bias_correction1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
