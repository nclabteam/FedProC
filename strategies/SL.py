"""
Selective Learning (SL) — dynamic dual-mask training strategy.

Paper: Selective Learning for Deep Time Series Forecasting
Venue: NeurIPS 2025
Link : https://openreview.net/forum?id=kgzRy6nD6D

The strategy applies two complementary masks to the per-timestep loss:

1. **Uncertainty mask (Mᵤ)** — tracks prediction residuals across a sliding
   window of epochs on CPU, computes differential entropy (approximated as
   variance for Gaussian residuals), and masks out the top ``r_u`` fraction
   of high-entropy timesteps.
2. **Anomaly mask (Mₐ)** — uses a lightweight auxiliary DLinear estimator
   (auto-bootstrapped for ``estimator_epochs``) to predict a residual lower
   bound.  Timesteps closest to that bound are masked out at ratio ``r_a``.

Both masks default to *None* (disabled); set ``--r_u`` and/or ``--r_a`` to
activate.  SL is **model-agnostic** and works with any registered forecaster.
"""

import time
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from .nFL import nFL, nFL_Client


# ---------------------------------------------------------------------------
# DataLoader wrapper — yields batches with an extra ``idx`` field
# ---------------------------------------------------------------------------
class _DataLoaderWithIndex:
    """Wrap an existing DataLoader so every batch includes dataset indices.

    If the underlying collate returns a tuple ``(batch_x, batch_y, …)`` an
    extra ``idx`` tensor is appended.  This wrapper deliberately bypasses
    ``num_workers`` to avoid pickling issues with the index injection; it
    reuses the original ``batch_sampler`` and ``collate_fn``.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        self._dataloader = dataloader
        self.dataset = dataloader.dataset
        self.collate_fn = dataloader.collate_fn or default_collate
        self.batch_sampler = dataloader.batch_sampler

    def __iter__(self):
        for batch_indices in self.batch_sampler:
            batch = [self.dataset[i] for i in batch_indices]
            collated = self.collate_fn(batch)
            idx_tensor = torch.tensor(batch_indices, dtype=torch.long)
            # FedProC loaders always return tuples (batch_x, batch_y, x_mark, y_mark)
            if isinstance(collated, (tuple, list)):
                yield (*collated, idx_tensor)
            elif isinstance(collated, dict):
                collated = dict(collated)
                collated["idx"] = idx_tensor
                yield collated
            else:
                yield collated, idx_tensor

    def __len__(self):
        return len(self._dataloader)

    def __getattr__(self, name):
        return getattr(self._dataloader, name)


# ---------------------------------------------------------------------------
# Lightweight DLinear estimator (self-contained, no external dependency)
# ---------------------------------------------------------------------------
class _SimpleMovingAvgDecomp(nn.Module):
    """Moving-average series decomposition."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x: torch.Tensor):
        # x: [B, L, D]
        trend = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal = x - trend
        return seasonal, trend


class _DLinearEstimator(nn.Module):
    """Minimal DLinear used as the anomaly-mask estimator.

    This is a self-contained re-implementation so SL does not depend on the
    registered ``models.DLinear`` (which requires the full model-init pipeline).
    """

    def __init__(self, input_len: int, output_len: int, channels: int) -> None:
        super().__init__()
        self.decomposition = _SimpleMovingAvgDecomp(kernel_size=25)
        self.linear_seasonal = nn.Linear(input_len, output_len)
        self.linear_trend = nn.Linear(input_len, output_len)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # x: [B, L, D]
        seasonal, trend = self.decomposition(x)
        seasonal = self.linear_seasonal(seasonal.permute(0, 2, 1))
        trend = self.linear_trend(trend.permute(0, 2, 1))
        return (seasonal + trend).permute(0, 2, 1)  # [B, output_len, D]


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
class SL(nFL):
    """
    Selective Learning — model-agnostic dual-mask training strategy.

    Inherits from ``nFL`` (no federation). Each client trains independently
    using uncertainty and/or anomaly masks to exclude noisy timesteps from
    the loss computation.
    """

    compulsory = {**nFL.compulsory}
    optional = {
        "r_u": None,
        "r_a": None,
        "estimator_epochs": 5,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument(
            "--r_u",
            type=float,
            default=None,
            help="Uncertainty masking ratio in (0, 1). None = disabled.",
        )
        parser.add_argument(
            "--r_a",
            type=float,
            default=None,
            help="Anomaly masking ratio in (0, 1). None = disabled.",
        )
        parser.add_argument(
            "--estimator_epochs",
            type=int,
            default=None,
            help="Epochs to pre-train the DLinear anomaly estimator.",
        )

    def evaluate_generalization_loss(self, *args, **kwargs):
        pass

    def _pre_eval_hook(self, dataset_type: str) -> None:
        self.evaluate_personalization_loss(dataset_type)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class SL_Client(nFL_Client):
    """Client-side Selective Learning logic."""

    def train(self) -> Dict[str, Any]:
        seed = self._loader_seed("train") if hasattr(self, "_loader_seed") else None
        self._set_worker_seed(seed)

        train_loader = self.load_train_data()
        start_time = time.time()

        r_u: Optional[float] = self.r_u
        r_a: Optional[float] = self.r_a
        estimator_epochs: int = self.estimator_epochs

        # Move model to device
        self.model.to(self.device)
        self._move_optimizer_state_to_param_devices(self.optimizer)

        # ---- index-aware loader ----
        idx_loader = _DataLoaderWithIndex(train_loader)
        num_samples = len(train_loader.dataset)

        # ---- state for uncertainty mask ----
        history_residual: Optional[torch.Tensor] = None
        uncertainty_mask: Optional[torch.Tensor] = None

        # ---- anomaly estimator ----
        estimator: Optional[_DLinearEstimator] = None
        if r_a is not None:
            estimator = _DLinearEstimator(
                input_len=self.input_len,
                output_len=self.output_len,
                channels=self.input_channels,
            ).to(self.device)
            self._pretrain_estimator(estimator, train_loader, estimator_epochs)

        # ---- main training loop ----
        for epoch in range(self.epochs):
            self.model.train()
            for batch_x, batch_y, x_mark, y_mark, idx in idx_loader:
                batch_x = batch_x.to(
                    device=self.device, dtype=torch.float32, non_blocking=True
                )
                batch_y = batch_y.to(
                    device=self.device, dtype=torch.float32, non_blocking=True
                )
                x_mark = x_mark.to(
                    device=self.device, dtype=torch.float32, non_blocking=True
                )
                y_mark = y_mark.to(
                    device=self.device, dtype=torch.float32, non_blocking=True
                )

                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
                residual = torch.abs(outputs - batch_y)

                # --- build combined mask (True = keep, False = discard) ---
                mask = torch.ones_like(batch_y, dtype=torch.bool)

                # Uncertainty mask
                if r_u is not None:
                    if history_residual is None:
                        _, output_len, num_features = batch_y.shape
                        history_residual = torch.empty(
                            (num_samples, output_len, num_features),
                            device="cpu",
                        )
                    # Update history on CPU
                    history_residual[idx] = residual.detach().cpu()
                    # Apply previous epoch's mask
                    if uncertainty_mask is not None:
                        expanded_idx = idx.unsqueeze(-1) + torch.arange(
                            self.output_len, device="cpu"
                        )
                        unc_mask = uncertainty_mask[expanded_idx].to(self.device)
                        mask = mask & unc_mask

                # Anomaly mask
                if r_a is not None and estimator is not None:
                    with torch.no_grad():
                        est_out = estimator(batch_x)
                    residual_lb = torch.abs(est_out - batch_y)
                    dist = residual - residual_lb
                    thresholds = torch.quantile(dist, r_a, dim=1, keepdim=True)
                    ano_mask = dist > thresholds
                    mask = mask & ano_mask

                # Masked loss — only penalise generalizable timesteps
                masked_outputs = outputs * mask
                masked_targets = batch_y * mask
                # Scale by kept fraction so gradient magnitude is stable
                kept = mask.sum().clamp(min=1)
                loss = self.loss(masked_outputs, masked_targets) * mask.numel() / kept
                loss.backward()
                self.optimizer.step()

            # End-of-epoch: recompute uncertainty mask for next epoch
            if r_u is not None and history_residual is not None:
                res_entropy = self._compute_entropy(history_residual)
                thresholds = torch.quantile(res_entropy, 1 - r_u, dim=0, keepdim=True)
                uncertainty_mask = res_entropy < thresholds  # [N+H-1, C]

            self.scheduler.step()

        # Offload model to CPU
        if self.efficiency != "high":
            self.model.to("cpu")

        return {
            "model": self.model,
            "optimizer_state": self.optimizer,
            "train_time": time.time() - start_time,
            "train_samples": self.train_samples,
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _pretrain_estimator(
        self,
        estimator: _DLinearEstimator,
        train_loader: DataLoader,
        epochs: int,
    ) -> None:
        """Bootstrap the DLinear anomaly estimator for a few epochs."""
        opt = torch.optim.Adam(estimator.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        estimator.train()
        for _ in range(epochs):
            for batch_x, batch_y, x_mark, y_mark in train_loader:
                batch_x = batch_x.to(
                    device=self.device, dtype=torch.float32, non_blocking=True
                )
                batch_y = batch_y.to(
                    device=self.device, dtype=torch.float32, non_blocking=True
                )
                opt.zero_grad(set_to_none=True)
                out = estimator(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                opt.step()
        estimator.eval()

    @staticmethod
    def _compute_entropy(residual: torch.Tensor) -> torch.Tensor:
        """Compute per-timestep residual entropy (variance proxy).

        Given a residual tensor of shape ``(N, H, C)`` where N is dataset size,
        H is the output/prediction length, and C is the number of features, this
        computes the variance of residuals along the *anti-diagonals* of the
        ``(sample, timestep)`` matrix — exactly following the NeurIPS 2025 paper.

        Returns:
            Tensor of shape ``(N + H - 1, C)`` — one entropy value per
            virtual timestep per feature.
        """
        num_samples, output_len, num_features = residual.shape

        # Diagonal indices: sample i, offset j → virtual timestep i + j
        ids = (
            torch.arange(num_samples, device=residual.device)[:, None]
            + torch.arange(output_len, device=residual.device)[None, :]
        )  # [N, H]

        x_flat = residual.view(-1, num_features)  # [N*H, C]
        ids_flat = ids.view(-1, 1).expand(-1, num_features)  # [N*H, C]

        result_shape = (num_samples + output_len - 1, num_features)
        sum_per_id = torch.zeros(
            result_shape, dtype=residual.dtype, device=residual.device
        )
        sum_sq_per_id = torch.zeros_like(sum_per_id)

        sum_per_id.scatter_add_(0, ids_flat, x_flat)
        sum_sq_per_id.scatter_add_(0, ids_flat, (residual**2).view(-1, num_features))

        counts = torch.bincount(
            ids.view(-1), minlength=num_samples + output_len - 1
        ).to(dtype=residual.dtype)
        counts = counts.unsqueeze(-1).expand(-1, num_features)

        mean = sum_per_id / counts
        var = (sum_sq_per_id / counts) - mean.pow(2)
        return var
