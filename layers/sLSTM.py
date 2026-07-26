import torch
import torch.nn as nn
import torch.nn.functional as F

from .CausalConv1d import CausalConv1d
from .GatedFeedForward import GatedFeedForward
from .LinearHeadwiseExpand import LinearHeadwiseExpand
from .MultiHeadLayerNorm import MultiHeadLayerNorm


class sLSTMCell(nn.Module):
    """Multi-head sLSTM cell (Beck et al., 2024, Sec. 2.2 and App. A.2).

    Scalar memory with an exponential input gate, a normalizer state ``n``
    that accumulates the input gate times all future forget gates, and the
    stabilizer state ``m`` of Eq. (15)-(17). Memory mixing runs through a
    block-diagonal recurrent kernel, so each head mixes only with its own past
    hidden state.

    Cell input activation is tanh and the hidden state activation is the
    identity, per App. A.2. Follows the reference NX-AI ``vanilla`` backend:
    the recurrent kernel is initialised to zeros, the forget-gate bias uses
    the block-dependent power-law schedule, and the gates are clamped to at
    most 1 for numerical safety.
    """

    gate_names = ("i", "f", "z", "o")

    def __init__(self, hidden_size, num_heads, block_idx=0, num_blocks=1):
        super().__init__()
        assert (
            hidden_size % num_heads == 0
        ), "hidden_size must be divisible by num_heads"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.block_idx = block_idx
        self.num_blocks = num_blocks

        n_gates = len(self.gate_names)
        self.recurrent_kernel = nn.Parameter(
            torch.zeros(num_heads, self.head_dim, n_gates, self.head_dim)
        )
        self.bias = nn.Parameter(torch.zeros(num_heads, n_gates, self.head_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.recurrent_kernel)
        with torch.no_grad():
            self.bias.zero_()
            # forget-gate bias: block-dependent power law, biasing early blocks
            # towards forgetting slowly and later blocks towards forgetting fast
            ratio = (
                self.block_idx / (self.num_blocks - 1) if self.num_blocks > 1 else 0.0
            )
            positions = torch.arange(self.head_dim, dtype=torch.float32)
            positions = positions / max(self.head_dim - 1, 1)
            f_index = self.gate_names.index("f")
            self.bias[:, f_index, :] = 5.0 - 12.0 * positions ** (0.3 + 1.3 * ratio)

    def forward(self, i_pre, f_pre, z_pre, o_pre):
        """Run the recurrence over a full sequence.

        Args:
            i_pre, f_pre, z_pre, o_pre: (batch, seq_len, hidden_size) gate
                pre-activations from the block's headwise projections.

        Returns:
            (batch, seq_len, hidden_size) hidden states, before GroupNorm.
        """
        batch_size, seq_len, _ = i_pre.shape
        heads, head_dim = self.num_heads, self.head_dim

        shape = (batch_size, seq_len, heads, head_dim)
        # stacked in gate_names order: i, f, z, o
        pre = torch.stack(
            [t.view(shape) for t in (i_pre, f_pre, z_pre, o_pre)], dim=3
        )  # (batch, seq_len, heads, gates, head_dim)
        pre = pre + self.bias

        h = i_pre.new_zeros(batch_size, heads, head_dim)
        c = i_pre.new_zeros(batch_size, heads, head_dim)
        n = i_pre.new_zeros(batch_size, heads, head_dim)
        m = i_pre.new_zeros(batch_size, heads, head_dim)

        outputs = []
        for t in range(seq_len):
            # memory mixing: the recurrence consumes the raw hidden state, not
            # the GroupNorm-ed block output
            recurrent = torch.einsum("bhd,hdge->bhge", h, self.recurrent_kernel)
            i_t, f_t, z_t, o_t = (pre[:, t] + recurrent).unbind(dim=2)

            log_f_plus_m = m + F.logsigmoid(f_t)
            if t == 0:
                # no history yet, so the stabilizer is just the input gate
                m = i_t
            else:
                m = torch.maximum(i_t, log_f_plus_m)
            i_gate = torch.exp(i_t - m).clamp(max=1.0)
            f_gate = torch.exp(log_f_plus_m - m).clamp(max=1.0)

            c = f_gate * c + i_gate * torch.tanh(z_t)
            n = f_gate * n + i_gate
            h = torch.sigmoid(o_t) * c / n
            outputs.append(h)

        stacked = torch.stack(outputs, dim=1)
        return stacked.reshape(batch_size, seq_len, self.hidden_size)


class sLSTMBlock(nn.Module):
    """Residual sLSTM block with post up-projection (Beck et al., 2024,
    Sec. 2.4 and Fig. 9) -- the Transformer-like arrangement the paper uses
    for every sLSTM block.

    pre-LayerNorm -> causal convolution with Swish feeding the input and
    forget gates (``z`` and ``o`` take the unconvolved input) -> headwise gate
    projections -> sLSTM cell -> dropout -> head-wise GroupNorm -> residual,
    then a GeLU-gated MLP -> residual.
    """

    def __init__(
        self,
        embedding_dim,
        num_heads,
        conv_kernel_size=4,
        ffn_proj_factor=1.3,
        ffn_act_fn="gelu",
        dropout=0.0,
        bias=False,
        block_idx=0,
        num_blocks=1,
        round_proj_to=64,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim, bias=False)
        self.conv1d = (
            CausalConv1d(embedding_dim, conv_kernel_size)
            if conv_kernel_size > 0
            else None
        )
        gate = lambda: LinearHeadwiseExpand(embedding_dim, num_heads, bias=bias)
        self.igate, self.fgate = gate(), gate()
        self.zgate, self.ogate = gate(), gate()
        for proj in (self.igate, self.fgate, self.zgate, self.ogate):
            proj.reset_parameters(dim=embedding_dim)

        self.cell = sLSTMCell(
            embedding_dim, num_heads, block_idx=block_idx, num_blocks=num_blocks
        )
        self.group_norm = MultiHeadLayerNorm(embedding_dim, num_heads)
        self.dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(embedding_dim, bias=False)
        self.ffn = GatedFeedForward(
            embedding_dim,
            proj_factor=ffn_proj_factor,
            act_fn=ffn_act_fn,
            dropout=dropout,
            bias=bias,
            num_blocks=num_blocks,
            round_proj_to=round_proj_to,
        )

    def forward(self, x):
        normed = self.norm(x)
        x_conv = (
            F.silu(self.conv1d(normed)) if self.conv1d is not None else normed
        )
        # the convolved branch drives only the input and forget gates
        y = self.cell(
            self.igate(x_conv),
            self.fgate(x_conv),
            self.zgate(normed),
            self.ogate(normed),
        )
        x = x + self.group_norm(self.dropout(y))
        return x + self.ffn(self.ffn_norm(x))
