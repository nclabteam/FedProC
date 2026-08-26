import torch
from torch import Tensor, nn

from .PositionalEncoding import PositionalEncoding
from .TSTEncoder import TSTEncoder


class TSTiEncoder(nn.Module):
    """Encode each variable's sequence of patches independently."""

    def __init__(
        self,
        patch_num: int,
        patch_len: int,
        n_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 16,
        d_k: int | None = None,
        d_v: int | None = None,
        d_ff: int = 256,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        act: str = "gelu",
        store_attn: bool = False,
        res_attention: bool = True,
        pre_norm: bool = False,
        pe: str | None = "zeros",
        learn_pe: bool = True,
    ) -> None:

        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len

        # Input encoding
        q_len = patch_num
        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.W_P = nn.Linear(patch_len, d_model)
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = PositionalEncoding(
            pe=pe,
            learn_pe=learn_pe,
            q_len=q_len,
            d_model=d_model,
        )

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(
            q_len=q_len,
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            pre_norm=pre_norm,
            activation=act,
            res_attention=res_attention,
            n_layers=n_layers,
            store_attn=store_attn,
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [bs x nvars x patch_len x patch_num]

        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0, 1, 3, 2)  # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)  # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # u: [bs * nvars x patch_num x d_model]
        u = self.dropout(u + self.W_pos)  # u: [bs * nvars x patch_num x d_model]

        # Encoder
        z = self.encoder(u)
        # z: [bs * nvars x patch_num x d_model]
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))
        # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0, 1, 3, 2)
        # z: [bs x nvars x d_model x patch_num]

        return z
