import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

optional = {
    "patch_size": 16,
    "d_model": 768,
    "stride": 8,
}


def args_update(parser):
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)


class PAttn(nn.Module):
    """
    Paper: https://arxiv.org/pdf/2406.16964
    Source: https://github.com/BennyTMT/LLMsForTimeSeries/blob/main/PAttn/models/PAttn.py
    """

    def __init__(self, configs):
        super(PAttn, self).__init__()
        self.seq_len = configs.input_len
        self.pred_len = configs.output_len
        self.patch_size = configs.patch_size
        self.stride = (
            configs.stride
            if hasattr(configs, "stride") and configs.stride is not None
            else configs.patch_size // 2
        )
        self.d_model = configs.d_model
        self.patch_num = (self.seq_len - self.patch_size) // self.stride + 2

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.in_layer = nn.Linear(self.patch_size, self.d_model)
        self.basic_attn = MultiHeadAttention(d_model=self.d_model)
        self.out_layer = nn.Linear(self.d_model * self.patch_num, self.pred_len)

    def norm(self, x, dim=1, means=None, stdev=None):
        """
        Normalization function
        For input [B, L, C], normalize along the sequence length dimension (dim=1)
        """
        if means is not None:
            # Denormalization
            return x * stdev + means
        else:
            # Normalization
            means = x.mean(dim, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(
                torch.var(x, dim=dim, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            x /= stdev
            return x, means, stdev

    def forward(self, x):
        # Input: [Batch, input_len, Channel] -> [Batch, Channel, input_len]
        x = x.transpose(1, 2)

        B, C = x.size(0), x.size(1)
        # Now: [Batch, Channel, input_len] e.g., [Batch, Channel, 336]

        # Normalize along sequence dimension
        x, means, stdev = self.norm(x, dim=2)

        # Apply padding: [Batch, Channel, input_len + stride]
        x = self.padding_patch_layer(x)

        # Create patches: [Batch, Channel, patch_num, patch_size]
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)

        # Project patches to d_model: [Batch, Channel, patch_num, d_model]
        x = self.in_layer(x)

        # Reshape for attention: [(Batch * Channel), patch_num, d_model]
        x = rearrange(x, "b c m l -> (b c) m l")

        # Apply multi-head attention
        x, _ = self.basic_attn(x, x, x)

        # Reshape back: [Batch, Channel, (patch_num * d_model)]
        x = rearrange(x, "(b c) m l -> b c (m l)", b=B, c=C)

        # Project to prediction length: [Batch, Channel, pred_len]
        x = self.out_layer(x)

        # Denormalize
        x = self.norm(x, means=means, stdev=stdev)

        # Transpose back to original format: [Batch, pred_len, Channel]
        x = x.transpose(1, 2)

        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, d_model=768, n_head=8, d_k=-1, d_v=-1, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        d_k = d_model // n_head if d_k == -1 else d_k
        d_v = d_k if d_v == -1 else d_v
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class Encoder_LLaTA(nn.Module):
    def __init__(self, input_dim, hidden_dim=768, num_heads=12, num_encoder_layers=1):
        super(Encoder_LLaTA, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

    def forward(self, x):
        # Input: [Batch, seq_len, input_dim]
        x = self.linear(x)
        # Transformer expects [seq_len, batch, hidden_dim]
        x = self.transformer_encoder(x.transpose(0, 1)).transpose(0, 1)
        # Output: [Batch, seq_len, hidden_dim]
        return x
