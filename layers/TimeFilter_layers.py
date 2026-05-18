import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.n_heads = n_heads

    def forward(self, adj, x):
        B, L, D = x.shape
        x = self.proj(x).view(B, L, self.n_heads, -1)
        adj = F.normalize(adj, p=1, dim=-1)
        x = torch.einsum("bhij,bjhd->bihd", adj, x).contiguous()
        x = x.view(B, L, -1)
        return x


class _MaskMoE(nn.Module):
    def __init__(self, n_vars, top_p=0.5, num_experts=3, in_dim=96):
        super().__init__()
        self.num_experts = num_experts
        self.n_vars = n_vars
        self.in_dim = in_dim
        self.gate = nn.Linear(self.in_dim, num_experts, bias=False)
        self.noise = nn.Linear(self.in_dim, num_experts, bias=False)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(2)
        self.top_p = top_p

    def _cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _cross_entropy(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return -torch.mul(x, torch.log(x + eps)).sum(dim=1).mean()

    def _noisy_top_k_gating(self, x, is_training, noise_epsilon=1e-2):
        clean_logits = self.gate(x)
        if is_training:
            noise_stddev = self.softplus(self.noise(x)) + noise_epsilon
            logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
        else:
            logits = clean_logits
        logits = self.softmax(logits)
        loss_dynamic = self._cross_entropy(logits)
        sorted_probs, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs > self.top_p
        threshold_indices = mask.long().argmax(dim=-1)
        threshold_mask = F.one_hot(
            threshold_indices, num_classes=sorted_indices.size(-1)
        ).bool()
        mask = mask & ~threshold_mask
        top_p_mask = torch.zeros_like(mask)
        zero_indices = (mask == 0).nonzero(as_tuple=True)
        top_p_mask[
            zero_indices[0],
            zero_indices[1],
            sorted_indices[zero_indices[0], zero_indices[1], zero_indices[2]],
        ] = 1
        sorted_probs = torch.where(mask, 0.0, sorted_probs)
        loss_importance = self._cv_squared(sorted_probs.sum(0))
        loss = loss_importance + 0.1 * loss_dynamic
        return top_p_mask, loss

    def forward(self, x, masks=None):
        B, H, L, _ = x.shape
        device = x.device
        dtype = torch.float32
        mask_base = torch.eye(L, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        if self.top_p == 0.0:
            return mask_base, 0.0
        x = x.reshape(B * H, L, L)
        gates, loss = self._noisy_top_k_gating(x, self.training)
        gates = gates.reshape(B, H, L, -1).float()
        if masks is None:
            N = L // self.n_vars
            masks = []
            for k in range(L):
                S = (
                    ((torch.arange(L) % N == k % N) & (torch.arange(L) != k))
                    .to(dtype)
                    .to(device)
                )
                T = (
                    (
                        (torch.arange(L) >= k // N * N)
                        & (torch.arange(L) < k // N * N + N)
                    )
                    .to(dtype)
                    .to(device)
                )
                ST = torch.ones(L).to(dtype).to(device) - S - T
                masks.append(torch.stack([S, T, ST], dim=0))
            masks = torch.stack(masks, dim=0)
        mask = torch.einsum("bhli,lid->bhld", gates, masks) + mask_base
        return mask, loss


def _mask_topk(x, alpha=0.5):
    k = int(alpha * x.shape[-1])
    _, topk_indices = torch.topk(x, k, dim=-1, largest=False)
    mask = torch.ones_like(x, dtype=torch.float32)
    mask.scatter_(-1, topk_indices, 0)
    return mask


class _GraphLearner(nn.Module):
    def __init__(self, dim, n_vars, top_p=0.5, in_dim=96):
        super().__init__()
        self.proj_1 = nn.Linear(dim, dim)
        self.proj_2 = nn.Linear(dim, dim)
        self.n_vars = n_vars
        self.mask_moe = _MaskMoE(n_vars, top_p=top_p, in_dim=in_dim)

    def forward(self, x, masks=None, alpha=0.5):
        adj = F.gelu(torch.einsum("bhid,bhjd->bhij", self.proj_1(x), self.proj_2(x)))
        adj = adj * _mask_topk(adj, alpha)
        mask, loss = self.mask_moe(adj, masks)
        adj = adj * mask
        return adj, loss


class _GraphFilter(nn.Module):
    def __init__(
        self, dim, n_vars, n_heads=4, scale=None, top_p=0.5, dropout=0.0, in_dim=96
    ):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.graph_learner = _GraphLearner(dim // n_heads, n_vars, top_p, in_dim=in_dim)
        self.graph_conv = GCN(dim, n_heads)

    def forward(self, x, masks=None, alpha=0.5):
        B, L, D = x.shape
        adj, loss = self.graph_learner(
            x.reshape(B, L, self.n_heads, -1).permute(0, 2, 1, 3), masks, alpha
        )
        adj = torch.softmax(adj, dim=-1)
        adj = self.dropout(adj)
        out = self.graph_conv(adj, x)
        return out, loss


class GraphBlock(nn.Module):
    def __init__(
        self, dim, n_vars, d_ff=None, n_heads=4, top_p=0.5, dropout=0.0, in_dim=96
    ):
        super().__init__()
        d_ff = dim * 4 if d_ff is None else d_ff
        self.gnn = _GraphFilter(
            dim, n_vars, n_heads, top_p=top_p, dropout=dropout, in_dim=in_dim
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, masks=None, alpha=0.5):
        out, loss = self.gnn(self.norm1(x), masks, alpha)
        x = x + out
        x = x + self.ffn(self.norm2(x))
        return x, loss


class TimeFilter_Backbone(nn.Module):
    def __init__(
        self,
        hidden_dim,
        n_vars,
        d_ff=None,
        n_heads=4,
        n_blocks=3,
        top_p=0.5,
        dropout=0.0,
        in_dim=96,
    ):
        super().__init__()
        d_ff = hidden_dim * 2 if d_ff is None else d_ff
        self.blocks = nn.ModuleList(
            [
                GraphBlock(hidden_dim, n_vars, d_ff, n_heads, top_p, dropout, in_dim)
                for _ in range(n_blocks)
            ]
        )
        self.n_blocks = n_blocks

    def forward(self, x, masks=None, alpha=0.5):
        moe_loss = 0.0
        for block in self.blocks:
            x, loss = block(x, masks, alpha)
            moe_loss += loss
        moe_loss /= self.n_blocks
        return x, moe_loss
