import math

import torch
import torch.nn as nn

from .PositionalEmbedding import PositionalEmbedding


class TokenEmbedding(nn.Module):
    """Project input features with a circular temporal convolution."""

    def __init__(self, c_in: int, d_model: int) -> None:
        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    """Embed discrete values with fixed sinusoidal vectors."""

    def __init__(self, c_in: int, d_model: int) -> None:
        super().__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """Combine calendar-field embeddings."""

    def __init__(
        self,
        d_model: int,
        embed_type: str = "fixed",
        freq: str = "h",
    ) -> None:
        super().__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        if freq == "t":
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.long()

        minute_x = (
            self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
        )
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    """Normalize raw calendar fields before a linear projection."""

    # Normalization constants per column: [month, day, weekday, hour, minute, second]
    _norm_max = torch.tensor([12.0, 31.0, 6.0, 23.0, 59.0, 59.0])

    def __init__(
        self,
        d_model: int,
        embed_type: str = "timeF",
        freq: str = "h",
    ) -> None:
        super().__init__()

        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        d_inp = freq_map[freq]
        self.d_inp = d_inp
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize raw integer features to [-0.5, 0.5]
        norm = self._norm_max[: self.d_inp].to(x.device, x.dtype)
        return self.embed(x / norm - 0.5)


class DataEmbedding(nn.Module):
    """Combine value, positional, and temporal embeddings."""

    def __init__(
        self,
        c_in: int,
        d_model: int,
        embed_type: str = "fixed",
        freq: str = "h",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor) -> torch.Tensor:
        x = (
            self.value_embedding(x=x)
            + self.temporal_embedding(x=x_mark)
            + self.position_embedding(x=x)
        )
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    """Combine value and optional temporal embeddings without positions."""

    def __init__(
        self,
        c_in: int,
        d_model: int,
        embed_type: str = "fixed",
        freq: str = "h",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor | None,
        x_mark: torch.Tensor | None,
    ) -> torch.Tensor:
        # https://github.com/huangst21/TimeKAN/blob/main/layers/Embed.py
        if x is None and x_mark is not None:
            return self.temporal_embedding(x=x_mark)
        if x_mark is None:
            x = self.value_embedding(x=x)
        else:
            x = self.value_embedding(x=x) + self.temporal_embedding(x=x_mark)
        return self.dropout(x)


class DataEmbedding_wo_pos_temp(nn.Module):
    """Embed values without positional or temporal features."""

    def __init__(
        self,
        c_in: int,
        d_model: int,
        embed_type: str = "fixed",
        freq: str = "h",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        x_mark: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.value_embedding(x=x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    """Embed variables as tokens, optionally with covariates."""

    def __init__(
        self,
        c_in: int,
        d_model: int,
        embed_type: str = "fixed",
        freq: str = "h",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        x_mark: torch.Tensor | None,
    ) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # [B, N, T]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        return self.dropout(x)


class DataEmbedding_wo_temp(nn.Module):
    """Combine value and positional embeddings without temporal features."""

    def __init__(
        self,
        c_in: int,
        d_model: int,
        embed_type: str = "fixed",
        freq: str = "h",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        x_mark: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.value_embedding(x=x) + self.position_embedding(x=x)
        return self.dropout(x)
