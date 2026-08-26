from argparse import Namespace

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class BaseScheduler(LRScheduler):
    """Keep every optimizer parameter group at its initial learning rate."""

    def __init__(
        self,
        optimizer: Optimizer,
        configs: Namespace,
        last_epoch: int = -1,
    ) -> None:
        super().__init__(optimizer=optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:
        return list(self.base_lrs)
