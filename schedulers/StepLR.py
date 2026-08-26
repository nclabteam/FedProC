from argparse import ArgumentParser, Namespace

from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR as TorchStepLR


class StepLR(TorchStepLR):
    """Adapt PyTorch step decay to FedProC configs."""

    optional = {"gamma": 0.5, "step_size": 1}

    @staticmethod
    def args_update(parser: ArgumentParser) -> None:
        parser.add_argument("--gamma", type=float, default=None)
        parser.add_argument("--step_size", type=int, default=None)

    def __init__(
        self,
        optimizer: Optimizer,
        configs: Namespace,
        last_epoch: int = -1,
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            step_size=configs.step_size,
            gamma=configs.gamma,
            last_epoch=last_epoch,
        )
