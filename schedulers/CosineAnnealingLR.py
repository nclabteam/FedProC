from argparse import ArgumentParser, Namespace

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR as TorchCosineAnnealingLR


class CosineAnnealingLR(TorchCosineAnnealingLR):
    """Adapt PyTorch cosine annealing to FedProC configs."""

    optional = {"eta_min": 0.0}

    @staticmethod
    def args_update(parser: ArgumentParser) -> None:
        parser.add_argument("--eta_min", type=float, default=None)

    def __init__(
        self,
        optimizer: Optimizer,
        configs: Namespace,
        last_epoch: int = -1,
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            T_max=configs.max_epochs,
            eta_min=configs.eta_min,
            last_epoch=last_epoch,
        )
