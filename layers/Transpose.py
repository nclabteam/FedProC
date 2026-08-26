import torch.nn as nn
from torch import Tensor


class Transpose(nn.Module):
    """Transpose configured dimensions and optionally make the result contiguous."""

    def __init__(self, *dims: int, contiguous: bool = False) -> None:
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x: Tensor) -> Tensor:
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)
