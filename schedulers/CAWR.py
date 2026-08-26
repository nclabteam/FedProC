from argparse import ArgumentParser, Namespace

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts as TorchCosineAnnealingWarmRestarts,
)


class CAWR(TorchCosineAnnealingWarmRestarts):
    """Adapt PyTorch cosine annealing with warm restarts to FedProC configs."""

    optional = {"T_0": 1, "T_mult": 1, "eta_min": 0.0}

    @staticmethod
    def args_update(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--T_0",
            type=int,
            default=None,
            help="Number of iterations for the first restart.",
        )
        parser.add_argument(
            "--T_mult",
            type=int,
            default=None,
            help="Multiplier for the next restart period.",
        )
        parser.add_argument(
            "--eta_min",
            type=float,
            default=None,
            help="Minimum learning rate.",
        )

    def __init__(
        self,
        optimizer: Optimizer,
        configs: Namespace,
        last_epoch: int = -1,
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            T_0=configs.T_0,
            T_mult=configs.T_mult,
            eta_min=configs.eta_min,
            last_epoch=last_epoch,
        )
