from argparse import ArgumentParser, Namespace

from torch.optim import Optimizer, lr_scheduler


class CALR(lr_scheduler.CosineAnnealingLR):
    """Adapt PyTorch cosine annealing to FedProC configs."""

    # Unset T_max means the mode's horizon.
    optional = {"eta_min": 0.0}

    @staticmethod
    def args_update(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--T_max",
            type=int,
            default=None,
            help="Steps per half cosine period (default: the mode's horizon).",
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
            T_max=getattr(configs, "T_max", configs.max_epochs),
            eta_min=configs.eta_min,
            last_epoch=last_epoch,
        )
