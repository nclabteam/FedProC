import torch
import torch.nn as nn
import torch.nn.functional as F

from .AutoAttention import AutoAttention
from .PositionalEncoding import PositionalEncoding


class DualAttention(nn.Module):
    """Decompose a series and model residual channel and temporal dependencies."""

    def __init__(
        self,
        enc_in: int,
        seq_len: int,
        d_model: int,
        dropout: float,
        pe_type: str | None,
        kernel_size: int,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.ld = LD(kernel_size=kernel_size)
        self.channel_attn_blocks = nn.ModuleList(
            [
                ChannelAttentionBlock(
                    enc_in=enc_in,
                    d_model=d_model,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.auto_attn_blocks = nn.ModuleList(
            [
                AutoAttentionBlock(
                    enc_in=enc_in,
                    d_model=d_model,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.position_embedder = DataEmbedding(
            pe_type=pe_type,
            seq_len=seq_len,
            d_model=d_model,
            c_in=enc_in,
        )

    def forward(self, inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.position_embedder(x=inp.permute(0, 2, 1)).permute(0, 2, 1)
        main = self.ld(inp=embedded)
        residual = embedded - main
        auto_residual = residual
        channel_residual = residual
        for auto_block, channel_block in zip(
            self.auto_attn_blocks,
            self.channel_attn_blocks,
        ):
            auto_residual = auto_block(residual=auto_residual)
            channel_residual = channel_block(residual=channel_residual)
        return auto_residual + channel_residual, main


class MultiHeadAttention(nn.Module):
    """Apply the Leddam multi-head attention projection."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 1,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.W_Q = nn.Linear(d_model, self.d_k * n_heads)
        self.W_K = nn.Linear(d_model, self.d_k * n_heads)
        self.W_V = nn.Linear(d_model, self.d_v * n_heads)
        self.sdp_attn = ScaledDotProductAttention(
            d_model=d_model,
            n_heads=n_heads,
            attn_dropout=attn_dropout,
        )
        self.to_out = nn.Sequential(
            nn.Linear(n_heads * self.d_v, d_model),
            nn.Dropout(proj_dropout),
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        batch_size = q.size(0)
        q_s = (
            self.W_Q(q)
            .view(
                batch_size,
                -1,
                self.n_heads,
                self.d_k,
            )
            .transpose(1, 2)
        )
        k_s = (
            self.W_K(q)
            .view(
                batch_size,
                -1,
                self.n_heads,
                self.d_k,
            )
            .permute(0, 2, 3, 1)
        )
        v_s = (
            self.W_V(q)
            .view(
                batch_size,
                -1,
                self.n_heads,
                self.d_v,
            )
            .transpose(1, 2)
        )

        output = self.sdp_attn(q=q_s, k=k_s, v=v_s)
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                -1,
                self.n_heads * self.d_v,
            )
        )
        return self.to_out(output)


class ScaledDotProductAttention(nn.Module):
    """Apply scaled dot-product attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(
            torch.tensor(head_dim**-0.5),
            requires_grad=False,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        attn_scores = torch.matmul(q, k) * self.scale
        attn_weights = self.attn_dropout(F.softmax(attn_scores, dim=-1))
        return torch.matmul(attn_weights, v)


class ChannelAttentionBlock(nn.Module):
    """Model dependencies between channels."""

    def __init__(self, enc_in: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.channel_att_norm = nn.BatchNorm1d(enc_in)
        self.fft_norm = nn.LayerNorm(d_model)
        self.channel_attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=1,
            proj_dropout=dropout,
        )
        self.fft_layer = nn.Sequential(
            nn.Linear(d_model, int(d_model * 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * 2), d_model),
        )

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        transposed = residual.permute(0, 2, 1)
        attended = self.channel_attn(q=transposed)
        normalized = self.channel_att_norm(attended + transposed)
        normalized = self.fft_norm(self.fft_layer(normalized) + normalized)
        return normalized.permute(0, 2, 1)


class AutoAttentionBlock(nn.Module):
    """Model periodic temporal dependencies."""

    def __init__(self, enc_in: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.auto_attn_norm = nn.BatchNorm1d(enc_in)
        self.fft_norm = nn.LayerNorm(d_model)
        self.auto_attn = AutoAttention(
            P=64,
            d_model=d_model,
            proj_dropout=dropout,
        )
        self.fft_layer = nn.Sequential(
            nn.Linear(d_model, int(d_model * 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * 2), d_model),
        )

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        normalized = self.auto_attn_norm(
            (self.auto_attn(inp=residual) + residual).permute(0, 2, 1)
        )
        normalized = self.fft_norm(self.fft_layer(normalized) + normalized)
        return normalized.permute(0, 2, 1)


class LD(nn.Module):
    """Extract a smooth local trend with one shared Gaussian kernel."""

    def __init__(self, kernel_size: int = 25) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            padding_mode="replicate",
            bias=True,
        )
        positions = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        weights = torch.exp(-((positions / 2) ** 2)).reshape(1, 1, -1)
        with torch.no_grad():
            self.conv.weight.copy_(F.softmax(weights, dim=-1))
            self.conv.bias.zero_()

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        channels_first = inp.permute(0, 2, 1)
        batch_size, channels, sequence_length = channels_first.shape
        output = self.conv(
            channels_first.reshape(batch_size * channels, 1, sequence_length)
        )
        return output.reshape(batch_size, channels, sequence_length).permute(0, 2, 1)


class DataEmbedding(nn.Module):
    """Embed each channel and add its learnable positional encoding."""

    def __init__(
        self,
        pe_type: str | None,
        seq_len: int,
        d_model: int,
        c_in: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.position_embedding = PositionalEncoding(
            pe=pe_type,
            learn_pe=True,
            q_len=c_in,
            d_model=d_model,
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.value_embedding(x) + self.position_embedding)
