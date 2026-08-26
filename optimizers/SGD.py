from argparse import ArgumentParser, Namespace

from torch.optim import SGD
from torch.optim.optimizer import ParamsT


class SGD(SGD):
    """Adapt PyTorch SGD to FedProC configuration objects."""

    optional = {
        "momentum": 0,
        "dampening": 0,
        "weight_decay": 0,
        "nesterov": False,
    }

    @classmethod
    def args_update(cls, parser: ArgumentParser) -> None:
        parser.add_argument("--momentum", type=float, default=None)
        parser.add_argument("--dampening", type=float, default=None)
        parser.add_argument("--weight_decay", type=float, default=None)
        parser.add_argument("--nesterov", default=None, action="store_true")

    def __init__(self, params: ParamsT, configs: Namespace) -> None:
        super().__init__(
            params=params,
            lr=configs.learning_rate,
            momentum=configs.momentum,
            dampening=configs.dampening,
            weight_decay=configs.weight_decay,
            nesterov=configs.nesterov,
        )
