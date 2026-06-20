from collections import OrderedDict

import torch

from .tFL import tFL, tFL_Client


class Elastic(tFL):

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

        # Per-layer sensitivity: weighted mean across clients
        sensitivities = torch.stack(
            [packages[cid]["sensitivity"] for cid in cids], dim=-1
        )
        agg_sensitivity = torch.sum(sensitivities * weights, dim=-1)
        max_sensitivity = sensitivities.max(dim=-1)[0]
        zeta = 1 + self.tau - agg_sensitivity / max_sensitivity.clamp(min=1e-12)

        # Elastic aggregation: new_global[i] = old[i] - zeta[i] * sum(w * (trained[i] - old[i]))
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
                server_p.float() - coef * (agg_trained - server_p.float())
            ).to(server_p.dtype)

        self._commit_global(new_global)


class Elastic_Client(tFL_Client):

    sample_ratio: float = 0.3
    mu: float = 0.95

    def fit(self) -> None:
        # Compute per-layer sensitivity on a sample of training data before training
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
        # Standard training
        super().fit()

    def package(self, train_time: float) -> dict:
        out = super().package(train_time)
        out["sensitivity"] = self._sensitivity
        return out
