import torch
import torch.nn as nn
import torch.nn.functional as F


class DishTS(nn.Module):
    """Normalize and denormalize series with learned statistics."""

    def __init__(
        self,
        num_features: int,
        seq_len: int,
        dish_init: str = "uniform",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.eps = eps
        if dish_init == "standard":
            self.reduce_mlayer = nn.Parameter(
                torch.rand(num_features, seq_len, 2) / seq_len
            )
        elif dish_init == "avg":
            self.reduce_mlayer = nn.Parameter(
                torch.ones(num_features, seq_len, 2) / seq_len
            )
        elif dish_init == "uniform":
            self.reduce_mlayer = nn.Parameter(
                torch.ones(num_features, seq_len, 2) / seq_len
                + torch.rand(num_features, seq_len, 2) / seq_len
            )
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str = "norm") -> torch.Tensor | None:
        if mode == "norm":
            self.preget(x)
            return self.normalize(x)
        elif mode == "denorm":
            return self.denormalize(x)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.phil) / torch.sqrt(self.xil + self.eps)
        x = x.mul(self.gamma) + self.beta
        return x

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        # x: B*H*D (forecasts)
        t1 = (x - self.beta) / self.gamma
        t2 = torch.sqrt(self.xih + self.eps)
        return t1 * t2 + self.phih

    def preget(self, x: torch.Tensor) -> None:
        # (B, T, N)
        x_transpose = x.permute(2, 0, 1)  # (N, B, T)
        theta = torch.bmm(x_transpose, self.reduce_mlayer).permute(1, 2, 0)
        theta = F.gelu(theta)
        self.phil, self.phih = theta[:, :1, :], theta[:, 1:, :]
        t = x.shape[1] - 1
        self.xil = torch.sum(torch.pow(x - self.phil, 2), axis=1, keepdim=True) / t
        self.xih = torch.sum(torch.pow(x - self.phih, 2), axis=1, keepdim=True) / t
