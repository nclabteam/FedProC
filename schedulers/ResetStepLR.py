from torch.optim.lr_scheduler import StepLR as TorchStepLR

# Default optional configs
optional = {
    "gamma": 0.5,
    "step_size": 1,
}


def args_update(parser):
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--step_size", type=int, default=None)


class ResetStepLR(TorchStepLR):
    def __init__(self, optimizer, configs, last_epoch=-1):
        self.optimizer = optimizer
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.reset_interval = getattr(configs, "epochs", None)
        super().__init__(
            optimizer=optimizer,
            step_size=configs.step_size,
            gamma=configs.gamma,
            last_epoch=last_epoch,
        )

    def step(self, epoch=None):
        """Perform a scheduler step, with LR reset."""
        super().step(epoch)
        # Print current learning rates for debugging
        lrs = [group["lr"] for group in self.optimizer.param_groups]
        # Reset LR after every reset_interval epochs
        if self.reset_interval is not None:
            current_epoch = self.last_epoch if epoch is None else epoch
            if (current_epoch + 1) % self.reset_interval == 0:
                for i, group in enumerate(self.optimizer.param_groups):
                    group["lr"] = self.base_lrs[i]
                self.last_epoch = -1  # Restart scheduler tracking
