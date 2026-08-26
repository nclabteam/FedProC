"""Selective Learning (SL) — dynamic dual-mask training strategy."""

from types import SimpleNamespace
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from models.DLinear import DLinear

from .nFL import nFL, nFL_Client


# ---------------------------------------------------------------------------
# DataLoader wrapper — yields batches with an extra ``idx`` field
# ---------------------------------------------------------------------------
class _DataLoaderWithIndex:
    """Wrap an existing DataLoader so every batch includes dataset indices."""

    def __init__(self, dataloader: DataLoader) -> None:
        self._dataloader = dataloader
        self.dataset = dataloader.dataset
        self.collate_fn = dataloader.collate_fn or default_collate
        self.batch_sampler = dataloader.batch_sampler

    def __iter__(self) -> Any:
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

    def __len__(self) -> Any:
        return len(self._dataloader)

    def __getattr__(self, name: Any) -> Any:
        return getattr(self._dataloader, name)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
class SL(nFL):
    """Selective Learning — model-agnostic dual-mask training strategy."""

    compulsory = {"sample_ratio": 1.0}
    optional = {
        "r_u": 0.3,
        "r_a": 0.3,
        "estimator_epochs": 100,
    }

    @classmethod
    def args_update(cls, parser: Any) -> None:
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


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class SL_Client(nFL_Client):
    """Client-side Selective Learning logic."""

    r_u: Optional[float] = 0.3
    r_a: Optional[float] = 0.3
    estimator_epochs: int = 100

    def fit(self) -> None:
        self._set_worker_seed(seed=self._loader_seed(dataset_type="train"))

        train_loader = self.load_train_data()
        self.initialize_scheduler(steps_per_epoch=len(train_loader))

        for name, ratio in (("r_u", self.r_u), ("r_a", self.r_a)):
            if ratio is not None and not 0 < ratio < 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.estimator_epochs <= 0:
            raise ValueError("estimator_epochs must be positive")

        # Move model to device
        self.model.to(self.device)
        self._move_optimizer_state_to_param_devices(optimizer=self.optimizer)

        # ---- index-aware loader ----
        idx_loader = _DataLoaderWithIndex(train_loader)
        num_samples = len(train_loader.dataset)

        # ---- state for uncertainty mask ----
        history_residual = self.auxiliary_state.get("sl_history_residual")
        expected_history_shape = (
            num_samples,
            self.output_len,
            self.output_channels,
        )
        if (
            history_residual is not None
            and tuple(history_residual.shape) != expected_history_shape
        ):
            history_residual = None
        uncertainty_mask = self.auxiliary_state.get("sl_uncertainty_mask")

        # ---- anomaly estimator ----
        estimator = None
        if self.r_a is not None:
            estimator = DLinear(
                SimpleNamespace(
                    input_len=self.input_len,
                    output_len=self.output_len,
                    moving_avg=25,
                    stride=1,
                )
            ).to(self.device)
            estimator_state = self.auxiliary_state.get("sl_estimator")
            if estimator_state:
                estimator.load_state_dict(estimator_state)
                estimator.eval()
            else:
                self._pretrain_estimator(
                    estimator=estimator,
                    train_loader=train_loader,
                    epochs=self.estimator_epochs,
                )

        # ---- main training loop ----
        for _ in range(self.epochs):
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
                signed_residual = batch_y - outputs
                residual = signed_residual.abs()

                # --- build combined mask (True = keep, False = discard) ---
                # Paper Eq. 3: M = M_u ∨ M_a (OR — discard only if BOTH say discard)
                unc_mask_batch: Optional[torch.Tensor] = None
                ano_mask_batch: Optional[torch.Tensor] = None

                # Uncertainty mask
                if self.r_u is not None:
                    if history_residual is None:
                        _, output_len, num_features = batch_y.shape
                        history_residual = torch.empty(
                            (num_samples, output_len, num_features),
                            device="cpu",
                        )
                    # Update history on CPU
                    history_residual[idx] = signed_residual.detach().cpu()
                    # Apply previous epoch's mask
                    if uncertainty_mask is not None:
                        expanded_idx = idx.unsqueeze(-1) + torch.arange(
                            self.output_len, device="cpu"
                        )
                        unc_mask_batch = uncertainty_mask[expanded_idx].to(self.device)

                # Anomaly mask
                if self.r_a is not None and estimator is not None:
                    with torch.no_grad():
                        est_out = estimator(batch_x)
                    residual_lb = torch.abs(est_out - batch_y)
                    dist = residual - residual_lb
                    thresholds = torch.quantile(dist, self.r_a, dim=1, keepdim=True)
                    ano_mask_batch = dist > thresholds

                # Combine: OR per paper (Eq. M = M_u ∨ M_a)
                if unc_mask_batch is not None and ano_mask_batch is not None:
                    mask = unc_mask_batch | ano_mask_batch
                elif unc_mask_batch is not None:
                    mask = unc_mask_batch
                elif ano_mask_batch is not None:
                    mask = ano_mask_batch
                else:
                    mask = torch.ones_like(batch_y, dtype=torch.bool)

                # Masked loss — only penalise generalizable timesteps
                kept = mask.sum().clamp(min=1)
                loss = (signed_residual.square() * mask).sum() / kept
                loss.backward()
                self.optimizer.step()
                self.step_scheduler_batch(
                    scheduler=self.scheduler,
                    batch_data=batch_x,
                )

            # End-of-epoch: recompute uncertainty mask for next epoch
            if self.r_u is not None and history_residual is not None:
                res_entropy = self._compute_entropy(residual=history_residual)
                thresholds = torch.quantile(
                    res_entropy, 1 - self.r_u, dim=0, keepdim=True
                )
                uncertainty_mask = res_entropy < thresholds  # [N+H-1, C]

            self.step_scheduler_epoch(scheduler=self.scheduler)

        if self.efficiency != "high":
            self.model.to("cpu")
        if history_residual is not None:
            self.auxiliary_state["sl_history_residual"] = history_residual
        if uncertainty_mask is not None:
            self.auxiliary_state["sl_uncertainty_mask"] = uncertainty_mask
        if estimator is not None:
            self.auxiliary_state["sl_estimator"] = {
                name: value.detach().cpu().clone()
                for name, value in estimator.state_dict().items()
            }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _pretrain_estimator(
        self,
        estimator: DLinear,
        train_loader: DataLoader,
        epochs: int,
    ) -> None:
        """Train the paper's DLinear residual-lower-bound estimator first."""
        opt = torch.optim.Adam(estimator.parameters(), lr=5e-4)
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
                loss = (out - batch_y).square().mean()
                loss.backward()
                opt.step()
        estimator.eval()

    @staticmethod
    def _compute_entropy(residual: torch.Tensor) -> torch.Tensor:
        """Compute per-timestep residual entropy (variance proxy)."""
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
        return ((sum_sq_per_id / counts) - mean.pow(2)).clamp_min_(0)
