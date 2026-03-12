from torch.optim.lr_scheduler import OneCycleLR

optional = {
    "max_lr": 0.1,
    "total_steps": None,
    "epochs": None,
    "steps_per_epoch": None,
    "pct_start": 0.3,
    "anneal_strategy": "cos",
    "cycle_momentum": True,
    "base_momentum": 0.85,
    "max_momentum": 0.95,
    "div_factor": 25.0,
    "final_div_factor": 10000.0,
    "three_phase": False,
}


def args_update(parser):
    parser.add_argument("--max_lr", type=float, default=None)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--pct_start", type=float, default=None)
    parser.add_argument("--anneal_strategy", type=str, default=None)
    parser.add_argument("--cycle_momentum", type=bool, default=None)
    parser.add_argument("--base_momentum", type=float, default=None)
    parser.add_argument("--max_momentum", type=float, default=None)
    parser.add_argument("--div_factor", type=float, default=None)
    parser.add_argument("--final_div_factor", type=float, default=None)
    parser.add_argument("--three_phase", type=bool, default=None)


class OneCycleLR(OneCycleLR):
    def __init__(self, optimizer, configs, last_epoch=-1):
        super().__init__(
            optimizer=optimizer,
            max_lr=configs.max_lr,
            total_steps=configs.total_steps,
            epochs=configs.epochs,
            steps_per_epoch=configs.steps_per_epoch,
            pct_start=configs.pct_start,
            anneal_strategy=configs.anneal_strategy,
            cycle_momentum=configs.cycle_momentum,
            base_momentum=configs.base_momentum,
            max_momentum=configs.max_momentum,
            div_factor=configs.div_factor,
            final_div_factor=configs.final_div_factor,
            three_phase=configs.three_phase,
            last_epoch=last_epoch,
        )
