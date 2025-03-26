import inspect

import torch

from .AdamW import AdamW, args_update, optional


class AdamWnanoGPT(AdamW):
    """
    Source:
        https://github.com/karpathy/nanoGPT/blob/9755682b981a45507f6eb9b11eadef8cb83cebd5/model.py#L263
        https://github.com/GestaltCogTeam/BasicTS/blob/master/basicts/runners/optim/optimizers.py
    """

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
