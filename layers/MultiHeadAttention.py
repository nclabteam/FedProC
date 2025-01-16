import torch.nn as nn

from .ScaledDotProductAttention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """
    Source:https://github.com/Levi-Ackman/Leddam/blob/main/layers/Leddam.py
    """

    def __init__(self, d_model, n_heads=1, attn_dropout=0.0, proj_dropout=0.2):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

        # Scaled Dot-Product Attention (multiple heads)
        self.sdp_attn = ScaledDotProductAttention(
            d_model, n_heads, attn_dropout=attn_dropout
        )
        # Poject output
        self.to_out = nn.Sequential(
            nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout)
        )

    def forward(self, Q, K=None, V=None, prev=None):
        bs = Q.size(0)
        if K is None:
            K = Q
        if V is None:
            V = Q
        # Linear (+ split in multiple heads)
        q_s = (
            self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        )  # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = (
            self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        )  # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = (
            self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)
        )  # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if prev is not None:
            output, prev = self.sdp_attn(q_s, k_s, v_s)
        else:
            output = self.sdp_attn(q_s, k_s, v_s)
        # output: [bs x n_heads x q_len x d_v]

        # back to the original inputs dimensions
        output = (
            output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        )  # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)
        if prev is not None:
            return output, prev
        else:
            return output
