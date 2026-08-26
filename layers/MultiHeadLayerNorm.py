import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadLayerNorm(nn.Module):
    """The GroupNorm of the xLSTM blocks: a head-wise LayerNorm, i.e. one
    LayerNorm over each head's channels, applied independently per time step
    (Beck et al., 2024, "xLSTM: Extended Long Short-Term Memory", Fig. 9 and
    Fig. 10).

    Follows the reference NX-AI implementation: the scale is parameterised
    residually as ``1 + weight`` with ``weight`` initialised to zero, and
    there is no bias by default.
    """

    def __init__(
        self,
        ndim: int,
        num_heads: int,
        weight: bool = True,
        bias: bool = False,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        assert ndim % num_heads == 0, "ndim must be divisible by num_heads"
        self.ndim = ndim
        self.num_heads = num_heads
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(ndim)) if weight else None
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    @property
    def weight_proxy(self) -> torch.Tensor | None:
        return None if self.weight is None else 1.0 + self.weight

    def reset_parameters(self) -> None:
        if self.weight is not None:
            nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *leading, ndim = x.shape
        # group_norm over (N, C) normalizes within each group of channels; all
        # leading dims are folded into N so no statistics are shared across them
        out = F.group_norm(
            x.reshape(-1, ndim),
            num_groups=self.num_heads,
            weight=self.weight_proxy,
            bias=self.bias,
            eps=self.eps,
        )
        return out.view(*leading, ndim)

    def extra_repr(self) -> str:
        return f"ndim={self.ndim}, num_heads={self.num_heads}, eps={self.eps}"
