from argparse import ArgumentParser, Namespace

from torch.optim import AdamW
from torch.optim.optimizer import ParamsT


class AdamW(AdamW):
    """Adapt PyTorch AdamW to FedProC configuration objects."""

    optional = {
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "weight_decay": 0.01,
        "amsgrad": False,
    }

    @classmethod
    def args_update(cls, parser: ArgumentParser) -> None:
        parser.add_argument("--beta1", type=float, default=None)
        parser.add_argument("--beta2", type=float, default=None)
        parser.add_argument("--epsilon", type=float, default=None)
        parser.add_argument("--weight_decay", type=float, default=None)
        parser.add_argument("--amsgrad", default=None, action="store_true")

    def __init__(self, params: ParamsT, configs: Namespace) -> None:
        super().__init__(
            params=params,
            lr=configs.learning_rate,
            betas=(configs.beta1, configs.beta2),
            eps=configs.epsilon,
            weight_decay=configs.weight_decay,
            amsgrad=configs.amsgrad,
        )
