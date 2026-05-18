import torch
from torch.optim import Optimizer


class TS_AdamW(Optimizer):
    """
    TS-AdamW: AdamW (decoupled weight decay) without second-moment bias correction.

    Applies the same heuristic as TS-Adam (bias_correction2 = 1) with
    decoupled weight decay following the AdamW formulation.

    Reference: "Rethinking Adam for Time Series Forecasting: A Simple
    Heuristic to Improve Optimization under Distribution Shifts",
    arXiv:2603.10095.
    """

    optional = {
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "weight_decay": 0.01,
        "amsgrad": False,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--beta1", type=float, default=None)
        parser.add_argument("--beta2", type=float, default=None)
        parser.add_argument("--epsilon", type=float, default=None)
        parser.add_argument("--weight_decay", type=float, default=None)
        parser.add_argument("--amsgrad", default=None, action="store_true")

    def __init__(self, params, configs):
        defaults = dict(
            lr=configs.learning_rate,
            betas=(configs.beta1, configs.beta2),
            eps=configs.epsilon,
            weight_decay=configs.weight_decay,
            amsgrad=configs.amsgrad,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            amsgrad = group["amsgrad"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Decoupled weight decay
                p.mul_(1 - group["lr"] * group["weight_decay"])

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"] / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
