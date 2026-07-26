import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .CausalConv1d import CausalConv1d
from .GatedFeedForward import round_proj_up_dim
from .LinearHeadwiseExpand import LinearHeadwiseExpand
from .MultiHeadLayerNorm import MultiHeadLayerNorm


class mLSTMCell(nn.Module):
    """Multi-head mLSTM cell with matrix memory (Beck et al., 2024, Sec. 2.3).

    The scalar cell state becomes a matrix ``C`` updated by the covariance
    rule, with the forget gate acting as decay rate and the input gate as
    learning rate. Retrieval divides by ``max(|n^T q|, ...)``, the normalizer
    dot product being lower bounded so it cannot vanish. Input and forget
    gates are scalars per head and are computed from the concatenation of
    q, k and v, as in the reference NX-AI implementation. mLSTM has no memory
    mixing, which is what makes the recurrence parallelizable; this is the
    recurrent form, equivalent to the parallel form of App. A.3.

    Note the ``max(|n^T q|, 1)`` bound of Eq. (21) is written for unstabilized
    states; here ``C`` and ``n`` carry the ``exp(-m)`` stabilizer factor, so
    the bound is scaled by the same factor to keep the two forms equal.
    """

    def __init__(self, embedding_dim, num_heads, eps=1e-6):
        super().__init__()
        assert (
            embedding_dim % num_heads == 0
        ), "embedding_dim must be divisible by num_heads"
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.eps = eps

        self.igate = nn.Linear(3 * embedding_dim, num_heads)
        self.fgate = nn.Linear(3 * embedding_dim, num_heads)
        self.outnorm = MultiHeadLayerNorm(embedding_dim, num_heads)
        self.reset_parameters()

    def reset_parameters(self):
        self.outnorm.reset_parameters()
        # gates start input-independent: the forget gate biased wide open on a
        # linear ramp, the input gate near zero
        nn.init.zeros_(self.fgate.weight)
        with torch.no_grad():
            self.fgate.bias.copy_(torch.linspace(3.0, 6.0, self.num_heads))
        nn.init.zeros_(self.igate.weight)
        nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)

    def forward(self, q, k, v):
        """Run the recurrence over a full sequence.

        Args:
            q, k, v: (batch, seq_len, embedding_dim) projections. Per Fig. 10
                q and k come from the convolved branch and v bypasses it.

        Returns:
            (batch, seq_len, embedding_dim) hidden states, GroupNorm applied,
            before the block's learnable skip and output gate.
        """
        batch_size, seq_len, _ = q.shape
        heads, head_dim = self.num_heads, self.head_dim

        if_input = torch.cat([q, k, v], dim=-1)
        i_pre = self.igate(if_input)  # (batch, seq_len, heads)
        f_pre = self.fgate(if_input)

        shape = (batch_size, seq_len, heads, head_dim)
        q = q.view(shape)
        k = k.view(shape) / math.sqrt(head_dim)
        v = v.view(shape)

        C = q.new_zeros(batch_size, heads, head_dim, head_dim)
        n = q.new_zeros(batch_size, heads, head_dim)
        m = q.new_zeros(batch_size, heads)

        outputs = []
        for t in range(seq_len):
            i_t, f_t = i_pre[:, t], f_pre[:, t]
            log_f_plus_m = m + F.logsigmoid(f_t)
            m = torch.maximum(i_t, log_f_plus_m)
            i_gate = torch.exp(i_t - m)[..., None, None]
            f_gate = torch.exp(log_f_plus_m - m)[..., None, None]

            q_t, k_t, v_t = q[:, t], k[:, t], v[:, t]
            C = f_gate * C + i_gate * torch.einsum("bhd,bhe->bhde", k_t, v_t)
            n = f_gate[..., 0] * n + i_gate[..., 0] * k_t

            retrieved = torch.einsum("bhd,bhde->bhe", q_t, C)
            dot = torch.einsum("bhd,bhd->bh", q_t, n).abs()
            denom = torch.maximum(dot, torch.exp(-m)) + self.eps
            outputs.append(retrieved / denom[..., None])

        h = torch.stack(outputs, dim=1).reshape(batch_size, seq_len, self.embedding_dim)
        return self.outnorm(h)


class mLSTMBlock(nn.Module):
    """Residual mLSTM block with pre up-projection (Beck et al., 2024,
    Sec. 2.4 and Fig. 10) -- the State-Space-Model-like arrangement the paper
    uses for every mLSTM block, because the matrix memory gains capacity in
    the higher-dimensional space.

    pre-LayerNorm -> up-projection by factor 2 into a cell branch and an
    externalized output gate -> dimension-wise causal convolution with Swish
    -> q and k from the convolved branch, v bypassing it -> mLSTM -> learnable
    skip added -> Swish output gate -> down-projection -> residual.

    There is no gated MLP afterwards: the up-projection plays that role.
    """

    def __init__(
        self,
        embedding_dim,
        num_heads,
        proj_factor=2.0,
        conv_kernel_size=4,
        qkv_proj_blocksize=4,
        dropout=0.0,
        bias=False,
        num_blocks=1,
        round_proj_to=64,
    ):
        super().__init__()
        inner = round_proj_up_dim(embedding_dim, proj_factor, round_proj_to)
        assert inner % num_heads == 0, (
            f"inner dim {inner} must be divisible by num_heads {num_heads}; "
            "adjust --embedding_dim, --proj_factor or --round_proj_to"
        )
        assert inner % qkv_proj_blocksize == 0, (
            f"inner dim {inner} must be divisible by "
            f"--qkv_proj_blocksize {qkv_proj_blocksize}"
        )
        self.inner = inner

        self.norm = MultiHeadLayerNorm(embedding_dim, num_heads=1)
        self.proj_up = nn.Linear(embedding_dim, 2 * inner, bias=bias)

        num_proj_heads = inner // qkv_proj_blocksize
        self.q_proj = LinearHeadwiseExpand(inner, num_proj_heads, bias=bias)
        self.k_proj = LinearHeadwiseExpand(inner, num_proj_heads, bias=bias)
        self.v_proj = LinearHeadwiseExpand(inner, num_proj_heads, bias=bias)

        self.conv1d = CausalConv1d(inner, conv_kernel_size)
        self.cell = mLSTMCell(inner, num_heads)
        self.learnable_skip = nn.Parameter(torch.ones(inner))
        self.proj_down = nn.Linear(inner, embedding_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters(embedding_dim, num_blocks)

    def reset_parameters(self, embedding_dim, num_blocks):
        # small init on the up-projection and on q/k/v, Wang init on the
        # down-projection -- all keyed to the outer embedding dim
        nn.init.normal_(self.proj_up.weight, std=math.sqrt(2 / (5 * embedding_dim)))
        nn.init.normal_(
            self.proj_down.weight, std=2 / num_blocks / math.sqrt(embedding_dim)
        )
        for layer in (self.proj_up, self.proj_down):
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        for proj in (self.q_proj, self.k_proj, self.v_proj):
            proj.reset_parameters(dim=embedding_dim)
        nn.init.ones_(self.learnable_skip)

    def forward(self, x):
        x_mlstm, z = self.proj_up(self.norm(x)).chunk(2, dim=-1)
        x_conv = F.silu(self.conv1d(x_mlstm))
        # q and k read the convolved branch; v is fed the unconvolved one
        h = self.cell(self.q_proj(x_conv), self.k_proj(x_conv), self.v_proj(x_mlstm))
        h = h + self.learnable_skip * x_conv
        h = h * F.silu(z)
        return x + self.dropout(self.proj_down(h))
