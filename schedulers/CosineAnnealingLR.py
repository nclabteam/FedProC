from torch.optim.lr_scheduler import CosineAnnealingLR


class CosineAnnealingLR(CosineAnnealingLR):

    optional = {
        "eta_min": 0.0,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--eta_min", type=float, default=None)


    def __init__(self, optimizer, configs, last_epoch=-1):
        super().__init__(
            optimizer=optimizer,
            T_max=configs.max_epochs,
            eta_min=configs.eta_min,
            last_epoch=last_epoch,
        )
