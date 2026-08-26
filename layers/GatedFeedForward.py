import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ACT_FN_REGISTRY = {
    "gelu": F.gelu,
    "relu": F.relu,
    "relu^2": lambda x: F.relu(x).square(),
    "sigmoid": F.sigmoid,
    "swish": F.silu,
    "selu": F.selu,
}


def round_proj_up_dim(
    embedding_dim: int,
    proj_factor: float,
    multiple_of: int = 64,
) -> int:
    """Up-projection dim, rounded up to a multiple (reference default: 64)."""
    if multiple_of <= 1:
        return int(round(embedding_dim * proj_factor))
    return int(math.ceil(embedding_dim * proj_factor / multiple_of) * multiple_of)


class GatedFeedForward(nn.Module):
    """Gated MLP of the post up-projection sLSTM block (Beck et al., 2024,
    "xLSTM: Extended Long Short-Term Memory", Fig. 9).

    GeLU-gated with projection factor 1.3 and no biases, matching the
    reference NX-AI implementation. The paper text quotes 4/3 for the same
    factor.
    """

    def __init__(
        self,
        embedding_dim: int,
        proj_factor: float = 1.3,
        act_fn: str = "gelu",
        dropout: float = 0.0,
        bias: bool = False,
        num_blocks: int = 1,
        round_proj_to: int = 64,
    ) -> None:
        super().__init__()
        assert act_fn in ACT_FN_REGISTRY, f"Unknown activation {act_fn!r}"
        self.inner = round_proj_up_dim(
            embedding_dim=embedding_dim,
            proj_factor=proj_factor,
            multiple_of=round_proj_to,
        )
        self.act_fn = ACT_FN_REGISTRY[act_fn]
        self.proj_up = nn.Linear(embedding_dim, 2 * self.inner, bias=bias)
        self.proj_down = nn.Linear(self.inner, embedding_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters(
            embedding_dim=embedding_dim,
            num_blocks=num_blocks,
        )

    def reset_parameters(self, embedding_dim: int, num_blocks: int) -> None:
        # small init on the up-projection, Wang init on the down-projection
        nn.init.normal_(self.proj_up.weight, std=math.sqrt(2 / (5 * embedding_dim)))
        nn.init.normal_(
            self.proj_down.weight, std=2 / num_blocks / math.sqrt(embedding_dim)
        )
        for layer in (self.proj_up, self.proj_down):
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_preact, up_proj = self.proj_up(x).split(self.inner, dim=-1)
        return self.dropout(self.proj_down(self.act_fn(gate_preact) * up_proj))
