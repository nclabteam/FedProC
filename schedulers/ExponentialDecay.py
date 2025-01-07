from torch.optim.lr_scheduler import _LRScheduler

optional = {
    "lr_gamma": 0.5,
    "decay_interval": 1,
}


def args_update(parser):
    parser.add_argument("--lr_gamma", type=float, default=None)
    parser.add_argument("--decay_interval", type=int, default=None)


class ExponentialDecay(_LRScheduler):
    def __init__(self, optimizer, configs, last_epoch=-1):
        """
        Exponential Decay Scheduler
        Args:
            optimizer: Wrapped optimizer.
            gamma: Multiplicative factor of decay (0.5 for halving).
            decay_interval: Interval (in epochs) to apply decay.
            last_epoch: The index of last epoch. Default: -1
        """
        self.gamma = configs.lr_gamma
        self.decay_interval = configs.decay_interval
        super(ExponentialDecay, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # Apply decay every decay_interval epochs
        factor = self.gamma ** ((self.last_epoch) // self.decay_interval)
        return [base_lr * factor for base_lr in self.base_lrs]
