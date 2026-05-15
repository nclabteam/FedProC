from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.SelfAttention_Family import TriangularCausalMask


class _FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous(), None


class _SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        d_keys = d_model // n_heads
        self.inner_attention = _FullAttention(attention_dropout=0.1)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


class Attention_Block(nn.Module):
    def __init__(self, d_model, d_ff=None, n_heads=8, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = _SelfAttention(d_model, n_heads)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, _ = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class _nconv(nn.Module):
    def forward(self, x, A):
        return torch.einsum("ncwl,vw->ncvl", (x, A)).contiguous()


class _mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super().__init__()
        self.nconv = _nconv()
        self.mlp = nn.Conv2d((gdep + 1) * c_in, c_out, kernel_size=(1, 1))
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0), device=x.device)
        d = adj.sum(1)
        a = adj / d.view(-1, 1)
        h = x
        out = [h]
        for _ in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)
        return self.mlp(torch.cat(out, dim=1))


class GraphBlock(nn.Module):
    def __init__(self, c_out, d_model, conv_channel, skip_channel, gcn_depth, dropout, propalpha, seq_len, node_dim):
        super().__init__()
        self.nodevec1 = nn.Parameter(torch.randn(c_out, node_dim))
        self.nodevec2 = nn.Parameter(torch.randn(node_dim, c_out))
        self.start_conv = nn.Conv2d(1, conv_channel, (d_model - c_out + 1, 1))
        self.gconv1 = _mixprop(conv_channel, skip_channel, gcn_depth, dropout, propalpha)
        self.gelu = nn.GELU()
        self.end_conv = nn.Conv2d(skip_channel, seq_len, (1, seq_len))
        self.linear = nn.Linear(c_out, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        out = x.unsqueeze(1).transpose(2, 3)
        out = self.start_conv(out)
        out = self.gelu(self.gconv1(out, adp))
        out = self.end_conv(out).squeeze(-1)
        out = self.linear(out)
        return self.norm(x + out)


class Predict(nn.Module):
    def __init__(self, individual, c_out, seq_len, pred_len, dropout):
        super().__init__()
        self.individual = individual
        self.c_out = c_out
        if individual:
            self.seq2pred = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(c_out)])
            self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(c_out)])
        else:
            self.seq2pred = nn.Linear(seq_len, pred_len)
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, c_out, seq_len]
        if self.individual:
            out = [self.dropout[i](self.seq2pred[i](x[:, i, :])) for i in range(self.c_out)]
            return torch.stack(out, dim=1)
        return self.dropout(self.seq2pred(x))
