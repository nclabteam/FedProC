from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

optional = {
    "T_0": 1,
    "T_mult": 1,
    "eta_min": 0.0,
}


def args_update(parser):
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
        "--eta_min", type=float, default=None, help="Minimum learning rate."
    )


class CAWR(CosineAnnealingWarmRestarts):
    def __init__(self, optimizer, configs, last_epoch=-1):
        super().__init__(
            optimizer=optimizer,
            T_0=configs.T_0,
            T_mult=configs.T_mult,
            eta_min=configs.eta_min,
            last_epoch=last_epoch,
        )
