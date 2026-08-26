import math

import numpy as np
import torch
import torch.nn as nn
from reformer_pytorch import LSHSelfAttention


class TriangularCausalMask:
    """Hold an upper-triangular causal attention mask."""

    def __init__(
        self,
        B: int,
        L: int,
        device: torch.device | str = "cpu",
    ) -> None:
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self) -> torch.Tensor:
        return self._mask


class ProbMask:
    """Hold the causal mask for selected probabilistic queries."""

    def __init__(
        self,
        B: int,
        H: int,
        L: int,
        index: torch.Tensor,
        scores: torch.Tensor,
        device: torch.device | str = "cpu",
    ) -> None:
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[
            torch.arange(B)[:, None, None],
            torch.arange(H)[None, :, None],
            index,
            :,
        ].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self) -> torch.Tensor:
        return self._mask


class FullAttention(nn.Module):
    """Apply full scaled dot-product attention."""

    def __init__(
        self,
        mask_flag: bool = True,
        factor: int = 5,
        scale: float | None = None,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: TriangularCausalMask | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / math.sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B=B, L=L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        if self.output_attention:
            return (V.contiguous(), A)
        return (V.contiguous(), None)


class ProbAttention(nn.Module):
    """Approximate attention by selecting sparse high-information queries."""

    def __init__(
        self,
        mask_flag: bool = True,
        factor: int = 5,
        scale: float | None = None,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
    ) -> None:
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        sample_k: int,
        n_top: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparsity measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[
            torch.arange(B)[:, None, None],
            torch.arange(H)[None, :, None],
            M_top,
            :,
        ]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(
        self,
        V: torch.Tensor,
        L_Q: int,
    ) -> torch.Tensor:
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            assert L_Q == L_V
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(
        self,
        context_in: torch.Tensor,
        V: torch.Tensor,
        scores: torch.Tensor,
        index: torch.Tensor,
        L_Q: int,
        attn_mask: ProbMask | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(
                B=B,
                H=H,
                L=L_Q,
                index=index,
                scores=scores,
                device=V.device,
            )
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)

        context_in[
            torch.arange(B)[:, None, None],
            torch.arange(H)[None, :, None],
            index,
            :,
        ] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = torch.ones([B, H, L_V, L_V]).type_as(attn).to(attn.device) / L_V
            attns[
                torch.arange(B)[:, None, None],
                torch.arange(H)[None, :, None],
                index,
                :,
            ] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: ProbMask | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * int(np.ceil(np.log(L_K)))
        u = self.factor * int(np.ceil(np.log(L_Q)))

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            Q=queries,
            K=keys,
            sample_k=U_part,
            n_top=u,
        )

        scale = self.scale or 1.0 / math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        context = self._get_initial_context(V=values, L_Q=L_Q)
        context, attn = self._update_context(
            context_in=context,
            V=values,
            scores=scores_top,
            index=index,
            L_Q=L_Q,
            attn_mask=attn_mask,
        )

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    """Project inputs around an attention implementation."""

    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        n_heads: int,
        d_keys: int | None = None,
        d_values: int | None = None,
    ) -> None:
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: TriangularCausalMask | ProbMask | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    """Adapt locality-sensitive hashing attention to the layer interface."""

    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        n_heads: int,
        d_keys: int | None = None,
        d_values: int | None = None,
        causal: bool = False,
        bucket_size: int = 4,
        n_hashes: int = 4,
    ) -> None:
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal,
        )

    def fit_length(self, queries: torch.Tensor) -> torch.Tensor:
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
        return torch.cat(
            [queries, torch.zeros(B, fill_len, C, device=queries.device)], dim=1
        )

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, None]:
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries=queries))[:, :N, :]
        return queries, None
