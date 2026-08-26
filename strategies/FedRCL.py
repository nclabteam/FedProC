from collections import OrderedDict
from typing import Any

import torch
import torch.nn.functional as F

from .tFL import tFL, tFL_Client


class FedRCL(tFL):
    """Federated Relaxed Contrastive Learning (Seo et al., CVPR 2024)."""

    optional = {
        "rcl_tau": 0.05,
        "rcl_beta": 1.0,
        "rcl_lambda": 0.7,
        "rcl_weight": 1.0,
        "rcl_num_classes": 4,
    }

    @classmethod
    def args_update(cls, parser: Any) -> None:
        parser.add_argument("--rcl_tau", type=float, default=None)
        parser.add_argument("--rcl_beta", type=float, default=None)
        parser.add_argument("--rcl_lambda", type=float, default=None)
        parser.add_argument("--rcl_weight", type=float, default=None)
        parser.add_argument("--rcl_num_classes", type=int, default=None)

    def aggregate_client_updates(self, packages: Any) -> None:
        new_global = OrderedDict(
            (
                name,
                torch.stack(
                    [
                        package["regular_model_params"][name]
                        for package in packages.values()
                    ]
                ).mean(dim=0),
            )
            for name in self.public_model_params
        )
        self._commit_global(new_params=new_global)


class FedRCL_Client(tFL_Client):
    """Train with relaxed contrastive regularization."""

    @staticmethod
    def compute_rcl_loss(
        features: Any, pseudo_labels: Any, tau: Any, beta: Any, lam: Any
    ) -> Any:
        """Compute relaxed contrastive loss."""
        B = features.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=features.device)

        features = F.normalize(features, dim=1)
        sim = features @ features.t()  # [B, B]

        device = features.device
        eye = torch.eye(B, device=device, dtype=torch.bool)
        labels_eq = pseudo_labels.unsqueeze(0) == pseudo_labels.unsqueeze(1)  # [B, B]
        pos_mask = labels_eq & ~eye  # same pseudo-class, not self

        if not pos_mask.any():
            return torch.tensor(0.0, device=device)

        logits = sim / tau  # [B, B]

        # SCL term per pair: -sim(i,j)/τ + log(Σ_{k≠i} exp(sim(i,k)/τ))
        log_denom = torch.logsumexp(
            logits.masked_fill(eye, float("-inf")), dim=1
        )  # [B]
        scl_per_pair = -logits + log_denom.unsqueeze(1)  # [B, B]

        # Divergence term (paper Eq. 5): β · 1_{j∈P(x_i)} · sim(i,j)/τ
        p_mask = pos_mask & (sim > lam)  # j ∈ P(x_i): same class, cos_sim > λ
        div_per_pair = beta * p_mask.float() * logits  # [B, B]

        # Average over all positive pairs
        total_per_pair = (scl_per_pair + div_per_pair) * pos_mask.float()
        return total_per_pair.sum() / pos_mask.sum().clamp(min=1)

    @staticmethod
    def assign_pseudo_labels(batch_y: Any, num_classes: Any) -> Any:
        """Assigns pseudo-class labels to TSF samples by binning target statistics."""
        # Compute per-sample statistic: mean across time and channels
        means = batch_y.mean(dim=(1, 2))  # [B]
        # Quantile-based binning for balanced classes
        quantiles = torch.linspace(0, 1, num_classes + 1, device=batch_y.device)
        boundaries = torch.quantile(means, quantiles[1:-1])
        labels = torch.bucketize(means, boundaries)
        return labels

    def train_one_epoch(
        self,
        model: Any,
        dataloader: Any,
        optimizer: Any,
        criterion: Any,
        scheduler: Any,
        device: Any,
        offload_after: Any = True,
    ) -> None:
        model.to(device)
        self._move_optimizer_state_to_param_devices(optimizer=optimizer)
        model.train()
        for batch_x, batch_y, x_mark, y_mark in dataloader:
            optimizer.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            x_mark = x_mark.to(device)
            y_mark = y_mark.to(device)

            # methodology.tex Algorithm 1, pseudocode step 10: task loss.
            outputs = model(batch_x, x_mark=x_mark, y_mark=y_mark)
            task_loss = criterion(outputs, batch_y)

            # methodology.tex Algorithm 1, pseudocode steps 8-9: L_RCL.
            # Use model outputs as representations (flattened)
            B = outputs.shape[0]
            features = outputs.reshape(B, -1)
            pseudo_labels = self.assign_pseudo_labels(
                batch_y=batch_y, num_classes=self.rcl_num_classes
            )

            # [methodology.tex, eq.6] — Relaxed contrastive loss
            rcl_loss = self.compute_rcl_loss(
                features=features,
                pseudo_labels=pseudo_labels,
                tau=self.rcl_tau,
                beta=self.rcl_beta,
                lam=self.rcl_lambda,
            )

            # methodology.tex Algorithm 1, pseudocode step 10: L = L_CE + L_RCL.
            loss = task_loss + self.rcl_weight * rcl_loss
            loss.backward()
            optimizer.step()
            self.step_scheduler_batch(
                scheduler=scheduler,
                batch_data=batch_x,
            )

        if offload_after:
            model.to("cpu")
        self.step_scheduler_epoch(scheduler=scheduler)

    def package(self) -> dict:
        package = super().package()
        package["__wire__"] = ("regular_model_params",)
        return package
