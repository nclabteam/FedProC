from .AdamW import AdamW


class AdamWnanoGPT(AdamW):

    optional = {
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "weight_decay": 0.01,
        "amsgrad": False,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--beta1", type=float, default=None)
        parser.add_argument("--beta2", type=float, default=None)
        parser.add_argument("--epsilon", type=float, default=None)
        parser.add_argument("--weight_decay", type=float, default=None)
        parser.add_argument("--amsgrad", default=None, action="store_true")

    def __init__(self, params, configs):
        params = [p for p in params if p.requires_grad]
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for p in params if p.dim() >= 2]
        nodecay_params = [p for p in params if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": configs.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        super().__init__(optim_groups, configs)
