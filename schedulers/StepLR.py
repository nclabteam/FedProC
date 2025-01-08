from torch.optim.lr_scheduler import StepLR

optional = {
    "gamma": 0.5,
    "step_size": 1,
}


def args_update(parser):
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--step_size", type=int, default=None)


class StepLR(StepLR):
    def __init__(self, optimizer, configs, last_epoch=-1):
        super().__init__(
            optimizer=optimizer,
            step_size=configs.step_size,
            gamma=configs.gamma,
            last_epoch=last_epoch,
        )
