from collections import OrderedDict

import torch

from .tFL import tFL, tFL_Client


class Elastic(tFL):
    """Elastic aggregation for federated learning (Liu et al., CVPR 2023).

    Aggregates client updates with per-parameter adaptive coefficients ζ^i based
    on parameter sensitivity. Sensitive parameters (large Ω^i) get ζ^i < 1 (restricted
    update), insensitive parameters get ζ^i > 1 (boosted update).

    Aggregation (Eq. 6-7):
        Ω^i  = Σ_k w_k * Ω_k^i                 (weighted sensitivity)
        ζ^i  = 1 + τ - Ω^i / max(Ω)
        θ_new = θ + ζ^i * (FedAvg(θ_k) - θ)   (elastic update)

    Sensitivity (Eq. 5): exponentially-decayed EMA of gradient norms per parameter.
    TSF adaptation: uses supervised-loss gradient instead of ||F(θ;x)||₂ gradient
    (unlabelled-data variant from the paper) since TSF outputs are numeric predictions
    rather than class logits.

    Default hyperparameters: τ = 0.5, μ = 0.95 (from the paper).
    Reference: arXiv:2301.01798. arXiv ID may differ; venue: CVPR 2023.
    """

    optional = {
        "tau": 0.5,
        "sample_ratio": 0.3,
        "mu": 0.95,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--tau", type=float, default=None)
        parser.add_argument("--sample_ratio", type=float, default=None)
        parser.add_argument("--mu", type=float, default=None)

    def aggregate_client_updates(self, packages) -> None:
        cids = list(packages.keys())
        scores = [packages[cid]["score"] for cid in cids]
        total = float(sum(scores))
        weights = torch.tensor([s / total for s in scores], dtype=torch.float32)

        # Per-parameter sensitivity: weighted mean across clients (Ω in Eq. 7)
        sensitivities = torch.stack(
            [packages[cid]["sensitivity"] for cid in cids], dim=-1
        )
        agg_sensitivity = torch.sum(sensitivities * weights, dim=-1)
        # Ω' = max(Ω) — global max across all layers (Eq. 7)
        max_sensitivity = agg_sensitivity.max().clamp(min=1e-12)
        # Adaptive coefficient ζ^i = 1 + τ - Ω^i / Ω'  (Eq. 7)
        zeta = 1 + self.tau - agg_sensitivity / max_sensitivity

        # Elastic update: θ_new = θ + ζ^i * (FedAvg(θ_k) - θ)  (Eq. 6)
        new_global = OrderedDict()
        for k, (name, server_p) in enumerate(self.public_model_params.items()):
            trained_stacked = torch.stack(
                [packages[cid]["regular_model_params"][name].float() for cid in cids],
                dim=-1,
            )
            agg_trained = torch.sum(
                trained_stacked * weights.to(trained_stacked.dtype), dim=-1
            )
            coef = zeta[k] if k < len(zeta) else torch.tensor(1.0)
            new_global[name] = (
                server_p.float() + coef * (agg_trained - server_p.float())
            ).to(server_p.dtype)

        self._commit_global(new_global)


class Elastic_Client(tFL_Client):

    sample_ratio: float = 0.3
    mu: float = 0.95

    def fit(self) -> None:
        # Compute per-parameter sensitivity on a sample of training data before training
        self.model.to(self.device)
        self.model.eval()
        sensitivity = torch.zeros(
            len(list(self.model.parameters())), device=self.device
        )
        sample_loader = self.load_data(
            file=self.train_file,
            sample_ratio=self.sample_ratio,
            shuffle=False,
            scaler=self.scaler,
            batch_size=self.batch_size,
            seed=self._loader_seed("train"),
        )
        for x, y, x_mark, y_mark in sample_loader:
            x = x.to(self.device, dtype=torch.float32)
            y = y.to(self.device, dtype=torch.float32)
            x_mark = x_mark.to(self.device, dtype=torch.float32)
            y_mark = y_mark.to(self.device, dtype=torch.float32)
            self.model.zero_grad()
            loss = self.loss(self.model(x, x_mark=x_mark, y_mark=y_mark), y)
            loss.backward()
            for i, param in enumerate(self.model.parameters()):
                if param.requires_grad and param.grad is not None:
                    sensitivity[i] = (
                        self.mu * sensitivity[i]
                        + (1 - self.mu) * (param.grad.data.norm() ** 2).abs()
                    )
                else:
                    sensitivity[i] = 1.0
        self._sensitivity = sensitivity.cpu()
        # Standard local training
        super().fit()

    def package(self) -> dict:
        out = super().package()
        out["sensitivity"] = self._sensitivity
        return out
