from torch.optim import AdamW

optional = {
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
    "weight_decay": 0.01,
    "amsgrad": False,
}


def args_update(parser):
    parser.add_argument("--beta1", type=float, default=None)
    parser.add_argument("--beta2", type=float, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--amsgrad", default=None, action="store_true")


class AdamW(AdamW):
    def __init__(self, params, configs):
        super().__init__(
            params=params,
            lr=configs.learning_rate,
            betas=(configs.beta1, configs.beta2),
            eps=configs.epsilon,
            weight_decay=configs.weight_decay,
            amsgrad=configs.amsgrad,
        )
