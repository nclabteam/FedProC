import copy
import math
from collections import OrderedDict

import numpy as np
import torch

from .tFL import tFL, tFL_Client
from .base import SharedMethods


class DeComFLShared(SharedMethods):
    """Shared utilities for DeComFL (zeroth-order FL)."""

    @staticmethod
    def generate_perturbation(dim: int, seed: int, device: str = "cpu") -> torch.Tensor:
        """Generate a Gaussian perturbation vector z ~ N(0, I_d) from a seed.

        Matches the paper's Algorithm 2 line 14 (Gaussian perturbations).
        """
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        return torch.randn(dim, device=device, generator=gen)

    @staticmethod
    def flatten_params(params: "OrderedDict[str, torch.Tensor]") -> torch.Tensor:
        """Flatten an OrderedDict of param tensors into a single 1-D tensor."""
        pieces = [p.data.detach().flatten().to(dtype=torch.float32) for p in params.values()]
        return torch.cat(pieces)

    @staticmethod
    def unflatten_params(
        flat: torch.Tensor,
        template: "OrderedDict[str, torch.Tensor]",
    ) -> "OrderedDict[str, torch.Tensor]":
        """Restore a flat 1-D tensor back into the shape of *template*."""
        result = OrderedDict()
        offset = 0
        for name, t in template.items():
            numel = t.numel()
            result[name] = flat[offset: offset + numel].reshape(t.shape).to(dtype=t.dtype)
            offset += numel
        return result


class DeComFL(DeComFLShared, tFL):
    """DeComFL: Dimension-Free Communication in FL via Zeroth-Order Optimization (paper 10020).

    Replaces per-round model/gradient upload with O(1) scalar communication.
    Server generates q random perturbation directions; clients evaluate loss at
    perturbed model weights; server reconstructs a ZO gradient estimate and updates
    the global model.

    Reference: arXiv:2405.15861.
    """

    optional = {
        "mu": 0.001,
        "q": 2,
        "zo_lr": 0.01,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--mu", type=float, default=None,
            help="Perturbation scale for ZO gradient estimation")
        parser.add_argument("--q", type=int, default=None,
            help="Number of random perturbation directions per round")
        parser.add_argument("--zo_lr", type=float, default=None,
            help="Learning rate for ZO gradient descent on server")
        return parser

    def __init__(self, configs, times):
        super().__init__(configs, times)
        self._perturbation_seeds: list = []
        self._round_counter = 0

    def package(self, client_id: int) -> dict:
        pkg = super().package(client_id)
        # Generate seeds once per round — all clients must share the same directions
        if not self._perturbation_seeds:
            self._perturbation_seeds = [
                int(np.random.randint(0, 2**31)) for _ in range(self.q)
            ]
        pkg["zo_seeds"] = self._perturbation_seeds
        pkg["zo_mu"] = self.mu
        pkg["zo_lr"] = self.zo_lr
        return pkg

    def aggregate_client_updates(self, packages: "OrderedDict[int, dict]") -> None:
        """Reconstruct ZO gradient from scalar g-values (paper Alg. 1 line 14-17).

        Server computes:  x_{r+1} = x_r - η · Σ_k (mean_i g_{i,r}^k) · z^k

        Each g_{i,r}^k = (f_i(x^k + μz^k) - f_i(x^k)) / μ is already a
        ZO gradient estimate — no additional finite-difference needed.
        """
        q = self.q
        eta = self.zo_lr
        num_clients = len(packages)

        if num_clients == 0:
            return

        # Generate perturbation vectors from seeds (shared across all clients)
        flat_dim = sum(p.numel() for p in self.public_model_params.values())
        device = "cpu"
        perturbations = []
        for seed in self._perturbation_seeds:
            u = self.generate_perturbation(flat_dim, seed, device)
            perturbations.append(u)

        # Accumulate ZO gradient:  g_avg · z  per direction
        zo_grad = torch.zeros(flat_dim, device=device)
        total_samples = 0

        for pkg in packages.values():
            g_scalars = pkg.get("zo_g_scalars", [])
            ci_samples = pkg.get("score", 1)

            for j in range(min(q, len(g_scalars))):
                # g_scalars[j] = (f(w^j + μz^j) - f(w^j)) / μ   (already the ZO gradient estimate)
                zo_grad += g_scalars[j] * perturbations[j] * ci_samples

            total_samples += ci_samples

        if total_samples > 0:
            zo_grad /= total_samples

        # SGD update: w ← w - η · g
        flat_w = self.flatten_params(self.public_model_params)
        flat_w = flat_w.to(device=device, dtype=torch.float32)
        flat_w = flat_w - eta * zo_grad
        updated = self.unflatten_params(flat_w, self.public_model_params)
        self._commit_global(updated)

        self._round_counter += 1


class DeComFL_Client(DeComFLShared, tFL_Client):
    mu: float = 0.001
    q: int = 2

    def set_parameters(self, package: dict) -> None:
        super().set_parameters(package)
        self._zo_seeds = package.get("zo_seeds", [])
        self._zo_mu = package.get("zo_mu", self.mu)
        self.zo_lr = package.get("zo_lr", 0.01)

    def fit(self) -> None:
        """K-step ZO-SGD local update with revert (paper Algorithm 2, lines 11-19).

        For each of K local steps:
          1. Generate perturbation z^k from seed
          2. Compute g^k = (f(w^k + μ·z^k) - f(w^k)) / μ
          3. w^{k+1} = w^k - η·g^k·z^k
        After all K steps, revert to the initial w^1.
        Uploads only the K gradient scalars {g^k} — O(1) uplink.
        """
        SharedMethods._set_worker_seed(self._loader_seed("train"))
        loader = self.load_train_data()

        self._zo_g_scalars = []

        # Snapshot initial model (will revert at the end)
        init_state = {
            k: self.model.state_dict()[k].data.clone() for k in self.regular_params_name
        }

        for seed in self._zo_seeds:
            flat_w = self.flatten_params(
                OrderedDict((k, self.model.state_dict()[k]) for k in self.regular_params_name)
            )
            device_flat = flat_w.device
            u = self.generate_perturbation(flat_w.shape[0], seed, str(device_flat))
            u = u.to(dtype=flat_w.dtype)

            # f(w) at current step
            f_w = float(np.mean(self._evaluate_on_loader(loader)))

            # f(w + μ·u)
            perturbed_flat = flat_w + self._zo_mu * u
            perturbed_params = self.unflatten_params(
                perturbed_flat,
                OrderedDict((k, self.model.state_dict()[k]) for k in self.regular_params_name),
            )
            self.model.load_state_dict(perturbed_params, strict=False)
            f_w_perturbed = float(np.mean(self._evaluate_on_loader(loader)))

            # ZO gradient scalar (forward difference)
            g_scalar = (f_w_perturbed - f_w) / self._zo_mu
            self._zo_g_scalars.append(g_scalar)

            # Local ZO-SGD step
            updated_flat = flat_w - self.zo_lr * g_scalar * u
            updated_params = self.unflatten_params(
                updated_flat,
                OrderedDict((k, self.model.state_dict()[k]) for k in self.regular_params_name),
            )
            self.model.load_state_dict(updated_params, strict=False)

        # Revert to initial state (paper Alg. 2, line 19)
        self.model.load_state_dict(init_state, strict=False)

    def _evaluate_on_loader(self, loader) -> list:
        """Compute per-batch losses on a dataloader without modifying model."""
        self.model.to(self.device)
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch_x, batch_y, x_mark, y_mark in loader:
                batch_x = batch_x.to(device=self.device, dtype=torch.float32)
                batch_y = batch_y.to(device=self.device, dtype=torch.float32)
                x_mark = x_mark.to(device=self.device, dtype=torch.float32)
                y_mark = y_mark.to(device=self.device, dtype=torch.float32)
                outputs = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
                loss = self.loss(outputs, batch_y)
                losses.append(loss.item())
        self.model.to("cpu")
        return losses

    def package(self) -> dict:
        """Return scalar gradient values instead of model parameters (O(1) uplink)."""
        pkg = {
            "__wire__": ("zo_g_scalars", "score"),
            "client_id": self.id,
            "regular_model_params": OrderedDict(),
            "personal_model_params": OrderedDict(),
            "optimizer_state": {},
            "scheduler_state": {},
            "score": self.train_samples,
            "zo_g_scalars": self._zo_g_scalars,
        }
        return pkg
