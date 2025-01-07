from torch.optim import SGD

optional = {
    "momentum": 0,
    "dampening": 0,
    "weight_decay": 0,
    "nesterov": False,
}


def args_update(parser):
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--dampening", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--nesterov", default=None, action="store_true")


class SGD(SGD):
    def __init__(self, params, configs):
        super(SGD, self).__init__(
            params=params,
            lr=configs.learning_rate,
            momentum=configs.momentum,
            dampening=configs.dampening,
            weight_decay=configs.weight_decay,
            nesterov=configs.nesterov,
        )
