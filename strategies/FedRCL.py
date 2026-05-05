import torch
import torch.nn.functional as F

from .base import Client, Server


class FedRCL(Server):
    """
    [methodology.tex, Algorithm 1] — Server side: standard FedAvg aggregation.
    No server-side changes needed.
    """

    optional = {
        "rcl_tau": 0.5,
        "rcl_beta": 1.0,
        "rcl_lambda": 0.5,
        "rcl_weight": 0.1,
        "rcl_num_classes": 4,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--rcl_tau", type=float, default=None)
        parser.add_argument("--rcl_beta", type=float, default=None)
        parser.add_argument("--rcl_lambda", type=float, default=None)
        parser.add_argument("--rcl_weight", type=float, default=None)
        parser.add_argument("--rcl_num_classes", type=int, default=None)


class FedRCL_Client(Client):
    """
    [methodology.tex, Algorithm 1] — Client side: L = L_task + rcl_weight * L_RCL.
    Overrides train_one_epoch to add the relaxed contrastive loss.
    """

    @staticmethod
    def compute_rcl_loss(features, pseudo_labels, tau, beta, lam):
        """
        Relaxed Contrastive Loss adapted for TSF.

        [methodology.tex, eq.6] — L_RCL(x_i, y_i; φ) =
          Σ_{j≠i, y_j=y_i} {
            -log(exp(⟨φ(x_i),φ(x_j)⟩/τ) / Σ_{k≠i} exp(⟨φ(x_i),φ(x_k)⟩/τ))
            + β · log(Σ_{x_k∈P(x_i)} exp(⟨φ(x_i),φ(x_k)⟩/τ) + exp(1/τ))
          }

        where P(x) = {x' | y_{x'}=y_x, ⟨φ(x'),φ(x)⟩ > λ}

        Args:
            features: [B, D] flattened representations.
            pseudo_labels: [B] integer pseudo-class labels.
            tau: temperature scalar.
            beta: divergence term weight.
            lam: similarity threshold for P(x).

        Returns:
            Scalar loss.
        """
        B = features.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=features.device)

        # [methodology.tex, eq.6] — cosine similarity ⟨φ(x_i), φ(x_j)⟩
        features = F.normalize(features, dim=1)
        sim = features @ features.t()  # [B, B]

        device = features.device
        eye = torch.eye(B, device=device, dtype=torch.bool)
        labels_eq = pseudo_labels.unsqueeze(0) == pseudo_labels.unsqueeze(1)  # [B, B]
        pos_mask = labels_eq & ~eye  # same pseudo-class, not self

        num_pos_per_anchor = pos_mask.sum(dim=1)  # [B]
        has_pos = num_pos_per_anchor > 0

        if not has_pos.any():
            return torch.tensor(0.0, device=device)

        # [methodology.tex, eq.6] — scaled logits
        logits = sim / tau  # [B, B]

        # Log-denominator: log(Σ_{k≠i} exp(sim(i,k)/τ))
        logits_for_denom = logits.masked_fill(eye, float("-inf"))
        log_denom = torch.logsumexp(logits_for_denom, dim=1)  # [B]

        # [methodology.tex, eq.6] — SCL term per pair: -sim(i,j)/τ + log_denom(i)
        scl_per_pair = -logits + log_denom.unsqueeze(1)  # [B, B]

        # [methodology.tex, eq.6] — Divergence term: P(x_i) = {same class, cos_sim > λ}
        p_mask = pos_mask & (sim > lam)  # [B, B]
        logits_p = logits.masked_fill(~p_mask, float("-inf"))  # [B, B]
        lse_p = torch.logsumexp(logits_p, dim=1)  # [B]
        exp_1_tau = 1.0 / tau
        # [methodology.tex, eq.6] — log(Σ_{k∈P} exp(sim/τ) + exp(1/τ))
        div_term = torch.logaddexp(lse_p, torch.tensor(exp_1_tau, device=device))  # [B]

        # Total per positive pair: scl + β·div
        total_per_pair = scl_per_pair + beta * div_term.unsqueeze(1)  # [B, B]

        # Average over all positive pairs
        loss = (total_per_pair * pos_mask.float()).sum()
        num_positives = pos_mask.sum()
        loss = loss / num_positives.clamp(min=1)

        return loss

    @staticmethod
    def assign_pseudo_labels(batch_y, num_classes):
        """
        Assigns pseudo-class labels to TSF samples by binning target statistics.

        Since TSF is regression with no class labels, we compute the per-sample
        mean of the target sequence and bin into quantile-based groups.

        Args:
            batch_y: [B, pred_len, channels] target tensor.
            num_classes: number of pseudo-class bins.

        Returns:
            [B] integer pseudo-labels.
        """
        # Compute per-sample statistic: mean across time and channels
        means = batch_y.mean(dim=(1, 2))  # [B]
        # Quantile-based binning for balanced classes
        quantiles = torch.linspace(0, 1, num_classes + 1, device=batch_y.device)
        boundaries = torch.quantile(means, quantiles[1:-1])
        labels = torch.bucketize(means, boundaries)
        return labels

    def train_one_epoch(
        self,
        model,
        dataloader,
        optimizer,
        criterion,
        scheduler,
        device,
        offload_after=True,
    ):
        model.to(device)
        self._move_optimizer_state_to_param_devices(optimizer)
        model.train()
        for batch in dataloader:
            optimizer.zero_grad()
            batch_x = batch[0].float().to(device)
            batch_y = batch[1].float().to(device)
            x_mark = batch[2].to(device) if len(batch) > 2 else None
            y_mark = batch[3].to(device) if len(batch) > 3 else None

            # [methodology.tex, Algorithm 1, line 10] — L_CE (task loss)
            outputs = model(batch_x, x_mark=x_mark, y_mark=y_mark)
            task_loss = criterion(outputs, batch_y)

            # [methodology.tex, Algorithm 1, line 8-9] — L_RCL
            # Use model outputs as representations (flattened)
            B = outputs.shape[0]
            features = outputs.reshape(B, -1)
            pseudo_labels = self.assign_pseudo_labels(batch_y, self.rcl_num_classes)

            # [methodology.tex, eq.6] — Relaxed contrastive loss
            rcl_loss = self.compute_rcl_loss(
                features=features,
                pseudo_labels=pseudo_labels,
                tau=self.rcl_tau,
                beta=self.rcl_beta,
                lam=self.rcl_lambda,
            )

            # [methodology.tex, Algorithm 1, line 10] — L = L_CE + L_RCL
            loss = task_loss + self.rcl_weight * rcl_loss
            loss.backward()
            optimizer.step()

        if offload_after:
            model.to("cpu")
        scheduler.step()
