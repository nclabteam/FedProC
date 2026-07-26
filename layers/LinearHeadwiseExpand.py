import math

import torch
import torch.nn as nn


class LinearHeadwiseExpand(nn.Module):
    """Block-diagonal ("headwise") linear projection.

    The input is split into ``num_heads`` contiguous groups and each group is
    projected independently, so the effective weight matrix is block-diagonal.
    Used for the sLSTM gate projections and the mLSTM q/k/v projections
    (Beck et al., 2024, "xLSTM: Extended Long Short-Term Memory", Fig. 9 and
    Fig. 10), matching ``LinearHeadwiseExpand`` in the reference NX-AI
    implementation with an expansion factor of 1.
    """

    def __init__(self, in_features, num_heads, bias=False):
        super().__init__()
        assert num_heads > 0, "num_heads must be positive"
        assert num_heads <= in_features, "num_heads must be <= in_features"
        assert (
            in_features % num_heads == 0
        ), "in_features must be a multiple of num_heads"

        self.in_features = in_features
        self.num_heads = num_heads
        block = in_features // num_heads
        self.weight = nn.Parameter(torch.empty(num_heads, block, block))
        self.bias = nn.Parameter(torch.zeros(in_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self, dim=None):
        """Small init of Nguyen & Salazar (2019): N(0, sqrt(2 / (5 * dim))).

        The reference implementation re-inits these with ``dim`` set to the
        block's outer embedding dim rather than the per-head block size, so
        callers pass it explicitly.
        """
        if dim is None:
            dim = self.weight.shape[-1]
        nn.init.normal_(self.weight, mean=0.0, std=math.sqrt(2 / (5 * dim)))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        shape = x.shape
        x = x.view(*shape[:-1], self.num_heads, -1)
        x = torch.einsum("...hd,hod->...ho", x, self.weight)
        x = x.reshape(*shape[:-1], -1)
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, num_heads={self.num_heads}, "
            f"bias={self.bias is not None}"
        )
