from torch.optim.lr_scheduler import _LRScheduler


class BaseScheduler(_LRScheduler):
    def __init__(self, optimizer, configs, last_epoch=-1):
        super(BaseScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # Return the base learning rates without modification
        return [base_lr for base_lr in self.base_lrs]
