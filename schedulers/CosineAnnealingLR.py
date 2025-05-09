from torch.optim.lr_scheduler import CosineAnnealingLR

optional = {
    "eta_min": 0.0,
}


def args_update(parser):
    parser.add_argument("--eta_min", type=float, default=None)


class CosineAnnealingLR(CosineAnnealingLR):
    def __init__(self, optimizer, configs, last_epoch=-1):
        super().__init__(
            optimizer=optimizer,
            T_max=configs.max_epochs,
            eta_min=configs.eta_min,
            last_epoch=last_epoch,
        )
