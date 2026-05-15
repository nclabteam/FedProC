import torch
import torch.nn as nn

from layers.PositionalEmbedding import PositionalEmbedding
from layers.StandardNorm import Normalize
from layers.TimeFilter_layers import TimeFilter_Backbone


class _PatchEmbed(nn.Module):
    def __init__(self, dim, patch_len, stride=None):
        super().__init__()
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride
        self.patch_proj = nn.Linear(self.patch_len, dim)
        self.pe = PositionalEmbedding(dim)

    def forward(self, x):
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = self.patch_proj(x) + self.pe(x)
        return x


class TimeFilter(nn.Module):
    """TimeFilter: Unified Graph-based Time Series Forecasting with Spatial-Temporal Filtering. AAAI 2025."""

    optional = {
        "d_model": 64,
        "d_ff": 128,
        "n_heads": 4,
        "e_layers": 3,
        "dropout": 0.1,
        "patch_len": 16,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--d_model", type=int, default=None)
        parser.add_argument("--d_ff", type=int, default=None)
        parser.add_argument("--n_heads", type=int, default=None)
        parser.add_argument("--e_layers", type=int, default=None)
        parser.add_argument("--dropout", type=float, default=None)
        parser.add_argument("--patch_len", type=int, default=None)

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.input_len
        self.pred_len = configs.output_len
        self.n_vars = configs.output_channels
        self.dim = configs.d_model
        self.patch_len = configs.patch_len
        self.stride = self.patch_len
        self.num_patches = int((self.seq_len - self.patch_len) / self.stride + 1)
        top_p = 0.5
        alpha = 0.1

        self._alpha = alpha
        self._L = self.seq_len * self.n_vars // self.patch_len

        self.norm = Normalize(configs.input_channels, affine=False)
        self.patch_embed = _PatchEmbed(self.dim, self.patch_len, self.stride)
        self.backbone = TimeFilter_Backbone(
            self.dim,
            self.n_vars,
            configs.d_ff,
            configs.n_heads,
            configs.e_layers,
            top_p,
            configs.dropout,
            in_dim=self._L,
        )
        self.head = nn.Linear(self.dim * self.num_patches, self.pred_len)

    def _get_mask(self, device):
        L = self._L
        N = self.seq_len // self.patch_len
        dtype = torch.float32
        masks = []
        for k in range(L):
            S = ((torch.arange(L) % N == k % N) & (torch.arange(L) != k)).to(dtype).to(device)
            T = (
                (torch.arange(L) >= k // N * N)
                & (torch.arange(L) < k // N * N + N)
                & (torch.arange(L) != k)
            ).to(dtype).to(device)
            ST = torch.ones(L, dtype=dtype, device=device) - S - T
            ST[k] = 0.0
            masks.append(torch.stack([S, T, ST], dim=0))
        return torch.stack(masks, dim=0)

    def forward(self, x, **kwargs):
        B, T, C = x.shape
        x = self.norm(x, "norm")
        x = x.permute(0, 2, 1).reshape(B, -1)  # [B, C*T] treat as single sequence
        x = self.patch_embed(x)  # [B, num_patches_total, D]
        x, _ = self.backbone(x, self._get_mask(x.device), self._alpha)
        x = x.reshape(B, self.n_vars, self.num_patches, self.dim)
        x = self.head(x.flatten(start_dim=-2))  # [B, n_vars, pred_len]
        x = x.permute(0, 2, 1)  # [B, pred_len, n_vars]
        x = self.norm(x, "denorm")
        return x
