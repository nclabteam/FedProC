from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from .base import SharedMethods
from .tFL import tFL, tFL_Client


class DeComFLShared(SharedMethods):
    """Shared utilities for DeComFL (zeroth-order FL)."""

    @staticmethod
    def generate_perturbation(dim: int, seed: int, device: str = "cpu") -> torch.Tensor:
        """Generate a Gaussian perturbation vector z ~ N(0, I_d) from a seed."""
        gen = torch.Generator(device=device)
        gen.manual_seed(seed * 17)
        return torch.randn(dim, device=device, generator=gen)

    @staticmethod
    def flatten_params(params: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Flatten an OrderedDict of param tensors into a single 1-D tensor."""
        pieces = [
            p.data.detach().flatten().to(dtype=torch.float32) for p in params.values()
        ]
        return torch.cat(pieces)

    @staticmethod
    def unflatten_params(
        vector: torch.Tensor,
        template: Mapping[str, torch.Tensor],
    ) -> "OrderedDict[str, torch.Tensor]":
        """Restore a flat 1-D tensor back into the shape of *template*."""
        result = OrderedDict()
        offset = 0
        for name, t in template.items():
            numel = t.numel()
            result[name] = (
                vector[offset : offset + numel].reshape(t.shape).to(dtype=t.dtype)
            )
            offset += numel
        return result


class DeComFL(DeComFLShared, tFL):
    """Dimension-free FL via zeroth-order optimization."""

    optional = {
        "mu": 0.001,
        "q": 2,
        "zo_lr": 0.01,
    }

    @classmethod
    def args_update(cls, parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "--mu",
            type=float,
            default=None,
            help="Perturbation scale for ZO gradient estimation",
        )
        parser.add_argument(
            "--q",
            type=int,
            default=None,
            help="Number of local ZO update steps K per round",
        )
        parser.add_argument(
            "--zo_lr",
            type=float,
            default=None,
            help="Learning rate for ZO gradient descent on server",
        )
        return parser

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        if self.mu <= 0 or self.q <= 0 or self.zo_lr <= 0:
            raise ValueError("DeComFL requires positive mu, q, and zo_lr")
        self._perturbation_seeds: list[int] = []

    def package(self, client_id: int) -> dict[str, Any]:
        pkg = super().package(client_id=client_id)
        # Paper Algorithm 1, pseudocode step 5: sample K shared seeds.
        if not self._perturbation_seeds:
            self._perturbation_seeds = [
                int(np.random.randint(0, 2**31)) for _ in range(self.q)
            ]
        pkg["zo_seeds"] = tuple(self._perturbation_seeds)
        pkg["__wire__"] += ("zo_seeds",)
        return pkg

    def aggregate_client_updates(
        self,
        packages: Mapping[int, dict[str, Any]],
    ) -> None:
        """Reconstruct the zeroth-order gradient."""
        if not packages:
            self._perturbation_seeds = []
            return

        flat_dim = sum(p.numel() for p in self.public_model_params.values())
        scalar_rows = []
        for package in packages.values():
            scalars = torch.as_tensor(package["zo_g_scalars"], dtype=torch.float32)
            if scalars.shape != (len(self._perturbation_seeds),):
                raise ValueError("DeComFL requires one scalar per shared seed")
            scalar_rows.append(scalars)

        # Paper Algorithm 1, pseudocode step 10: uniform client mean.
        global_scalars = torch.stack(scalar_rows).mean(dim=0)
        perturbations = torch.stack(
            [
                self.generate_perturbation(
                    dim=flat_dim,
                    seed=seed,
                    device="cpu",
                )
                for seed in self._perturbation_seeds
            ]
        )
        zo_grad = torch.sum(global_scalars.unsqueeze(dim=1) * perturbations, dim=0)

        # Paper Algorithm 1, pseudocode step 12: x_(r+1) = x_r - eta * sum_k(g_k z_k).
        flat_w = self.flatten_params(params=self.public_model_params)
        flat_w = flat_w.to(dtype=torch.float32) - self.zo_lr * zo_grad
        updated = self.unflatten_params(
            vector=flat_w,
            template=self.public_model_params,
        )
        self._commit_global(new_params=updated)
        self._perturbation_seeds = []


class DeComFL_Client(DeComFLShared, tFL_Client):
    def set_parameters(self, package: dict[str, Any]) -> None:
        super().set_parameters(package=package)
        self._zo_seeds = list(package["zo_seeds"])

    def _batch_loss(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        x_mark: torch.Tensor,
        y_mark: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
        return self.loss(outputs, batch_y)

    def fit(self) -> None:
        """Run local zeroth-order optimization."""
        self._set_worker_seed(seed=self._loader_seed(dataset_type="train"))
        loader = self.load_train_data()
        self._zo_g_scalars = []
        self.model.to(self.device)
        self.model.eval()
        state = self.model.state_dict()
        initial_state = OrderedDict(
            (name, state[name].detach().clone()) for name in self.regular_params_name
        )
        batches = iter(loader)
        try:
            # Paper Algorithm 2, pseudocode steps 12-15: K sequential ZO updates.
            with torch.no_grad():
                for seed in self._zo_seeds:
                    try:
                        batch_x, batch_y, x_mark, y_mark = next(batches)
                    except StopIteration:
                        batches = iter(loader)
                        batch_x, batch_y, x_mark, y_mark = next(batches)

                    batch_x = batch_x.to(device=self.device, dtype=torch.float32)
                    batch_y = batch_y.to(device=self.device, dtype=torch.float32)
                    x_mark = x_mark.to(device=self.device, dtype=torch.float32)
                    y_mark = y_mark.to(device=self.device, dtype=torch.float32)

                    state = self.model.state_dict()
                    template = OrderedDict(
                        (name, state[name]) for name in self.regular_params_name
                    )
                    flat_w = self.flatten_params(params=template)
                    perturbation = self.generate_perturbation(
                        dim=flat_w.numel(),
                        seed=seed,
                        device=str(flat_w.device),
                    ).to(dtype=flat_w.dtype)
                    base_loss = self._batch_loss(
                        batch_x=batch_x,
                        batch_y=batch_y,
                        x_mark=x_mark,
                        y_mark=y_mark,
                    )
                    perturbed = self.unflatten_params(
                        vector=flat_w + self.mu * perturbation,
                        template=template,
                    )
                    self.model.load_state_dict(perturbed, strict=False)
                    perturbed_loss = self._batch_loss(
                        batch_x=batch_x,
                        batch_y=batch_y,
                        x_mark=x_mark,
                        y_mark=y_mark,
                    )

                    # Paper Eq. 3 / Algorithm 2, pseudocode step 14: forward difference.
                    scalar = (perturbed_loss - base_loss) / self.mu
                    self._zo_g_scalars.append(float(scalar))
                    updated = self.unflatten_params(
                        vector=flat_w - self.zo_lr * scalar * perturbation,
                        template=template,
                    )
                    self.model.load_state_dict(updated, strict=False)
        finally:
            # Paper Algorithm 2, pseudocode step 17: revert to x_(i,r)^1.
            self.model.load_state_dict(initial_state, strict=False)
            if self.efficiency != "high":
                self.model.to("cpu")

    def package(self) -> dict[str, Any]:
        """Return scalar gradient values instead of model parameters (O(1) uplink)."""
        return {
            "__wire__": ("zo_g_scalars",),
            "client_id": self.id,
            "regular_model_params": OrderedDict(),
            "personal_model_params": OrderedDict(),
            "optimizer_state": {},
            "scheduler_state": {},
            "score": self.train_samples,
            "zo_g_scalars": self._zo_g_scalars,
        }
