"""MOTAR: Advantage-fitness Federated Evolution Strategies.

Each FL client = one ES population member. Client generates a random low-rank
perturbation (A_i, B_i), computes advantage fitness
  f_i^adv = L_i(M + Δ_i; B) - L_i(M + Δ_i + σ_g·E_i; B)
on the SAME mini-batch B, and uploads (f_i^adv, A_i, B_i) — O((m+n)·r) per
layer, no seed coordination between isolated systems.

Server reconstructs E_i = (1/√r) A_i B_i^T locally and updates
  M ← M + α_g · Σ k_i · f_i^adv · E_i
No backpropagation on clients. Δ_i is client-only, never communicated.

Reference: ideas/proposed/MOTAR/math.md
"""

import math
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .pFL import pFL, pFL_Client
from .tFL import tFL

# ─── hyperparameters ─────────────────────────────────────────────────────────


# ─── perturbation utilities ──────────────────────────────────────────────────


def _make_lora_factors(
    named_params, rank: int
) -> Tuple[Dict[str, np.ndarray], Dict[str, Optional[np.ndarray]]]:
    """Sample random low-rank factors (A, B) per parameter. No seed needed.

    For 2-D+ params: A ∈ R^{m×r}, B ∈ R^{n×r}, where n = prod(shape[1:]).
    For 1-D params (bias, LayerNorm): A is a vector of shape, B is None.
    Client sends these to the server; server reconstructs E = (1/√r) A Bᵀ.
    """
    A_factors: Dict[str, np.ndarray] = {}
    B_factors: Dict[str, Optional[np.ndarray]] = {}
    for name, param in named_params:
        shape = tuple(param.shape)
        if len(shape) >= 2:
            m = shape[0]
            n = int(np.prod(shape[1:]))
            A_factors[name] = np.random.standard_normal((m, rank)).astype(np.float32)
            B_factors[name] = np.random.standard_normal((n, rank)).astype(np.float32)
        else:
            A_factors[name] = np.random.standard_normal(shape).astype(np.float32)
            B_factors[name] = None
    return A_factors, B_factors


def _lora_to_dict(
    A_factors: Dict[str, np.ndarray],
    B_factors: Dict[str, Optional[np.ndarray]],
    named_params,
    rank: int,
    sigma: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Compute {name: sigma * E} in original param shapes.

    sigma=1.0 → pure perturbation direction, used as the update step.
    sigma=σ_g/σ_l → scaled, used only inside _eval_loss for the forward pass.
    Keeping these separate avoids double-applying σ (σ in fitness + σ in update = σ² effective step).
    """
    shapes = {name: tuple(param.shape) for name, param in named_params}
    result = {}
    for name, A in A_factors.items():
        B = B_factors.get(name)
        shape = shapes[name]
        if B is not None:
            E = (A @ B.T) / np.sqrt(rank)
            result[name] = (sigma * E).reshape(shape)
        else:
            result[name] = sigma * A
    return result


def _eval_loss(
    model: torch.nn.Module,
    delta: Dict[str, torch.Tensor],
    E_scaled: Optional[Dict[str, np.ndarray]],
    batch,
    criterion,
    device,
) -> float:
    """Forward-only loss of (model + delta + E_scaled) on a single batch.

    Temporarily adds offsets to params, evaluates, restores via try/finally.
    E_scaled=None → baseline L(M + Δ_i; B).
    """
    model.to(device)
    model.eval()
    loss_val = float("nan")
    try:
        with torch.no_grad():
            for name, param in model.named_parameters():
                d = delta.get(name)
                e = E_scaled.get(name) if E_scaled is not None else None
                if d is not None:
                    param.data.add_(d.to(device))
                if e is not None:
                    param.data.add_(torch.from_numpy(e).to(device))
            batch_x, batch_y, x_mark, y_mark = batch
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.float32)
            x_mark = x_mark.to(device=device, dtype=torch.float32)
            y_mark = y_mark.to(device=device, dtype=torch.float32)
            loss_val = criterion(
                model(batch_x, x_mark=x_mark, y_mark=y_mark), batch_y
            ).item()
    finally:
        with torch.no_grad():
            for name, param in model.named_parameters():
                d = delta.get(name)
                e = E_scaled.get(name) if E_scaled is not None else None
                if d is not None:
                    param.data.sub_(d.to(device))
                if e is not None:
                    param.data.sub_(torch.from_numpy(e).to(device))
    return loss_val


# ─── server ───────────────────────────────��────────────────────────────��─────


class MOTAR(pFL):

    optional = {
        "motar_rank": 4,  # r: rank of low-rank perturbations
        "motar_sigma_g": 1e-4,  # σ_g: global perturbation scale
        "motar_sigma_l": 1e-3,  # σ_l: local adapter perturbation scale
        "motar_K": 3,  # K: inner ES steps per round (Step A)
        "motar_alpha_g": 1.0,  # α_g: global model learning rate
        "motar_alpha_l": 1.0,  # α_l: local adapter learning rate
    }

    compulsory = {
        "exclude_server_model_processes": True,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--motar_rank", type=int, default=None)
        parser.add_argument("--motar_sigma_g", type=float, default=None)
        parser.add_argument("--motar_sigma_l", type=float, default=None)
        parser.add_argument("--motar_K", type=int, default=None)
        parser.add_argument("--motar_alpha_g", type=float, default=None)
        parser.add_argument("--motar_alpha_l", type=float, default=None)
        return parser

    _KNOWN_PACKAGE_KEYS = tFL._KNOWN_PACKAGE_KEYS | frozenset(
        {"fitness", "A_factors", "B_factors", "delta"}
    )

    def _apply_client_result(self, client, package: Optional[Dict[str, Any]]) -> None:
        """Apply MOTAR's training package back to *client*.

        Extends the base write-back with the per-client evolution-strategy state
        (advantage fitness, low-rank factors, and personalised adapter ``delta``)
        produced during local training so it survives Ray-parallel execution.

        Parameters
        ----------
        client : MOTAR_Client
            The original (non-Ray) client object to update.
        package : dict or None
            Return value of ``client.train()``.  ``None`` is a no-op.
        """
        if package is None:
            return
        super()._apply_client_result(client, package)
        client.delta = package["delta"]
        client._fitness = package["fitness"]
        client._A_factors = package["A_factors"]
        client._B_factors = package["B_factors"]

    def aggregate_models(self) -> None:
        """EGGROLL update: M ← M + α_g · Σ k_i · f_i^adv · (1/√r) A_i B_i^T"""
        with torch.no_grad():
            for client_info, weight in zip(self.client_data, self.weights):
                fitness = float(client_info["fitness"])
                coeff = self.motar_alpha_g * float(weight) * fitness
                A_factors = client_info["A_factors"]
                B_factors = client_info["B_factors"]

                for name, param in self.model.named_parameters():
                    A = A_factors.get(name)
                    if A is None:
                        continue
                    B = B_factors.get(name)
                    shape = tuple(param.shape)
                    if B is not None:
                        E = (A @ B.T) / np.sqrt(self.motar_rank)
                        update = torch.from_numpy(E.reshape(shape))
                    else:
                        update = torch.from_numpy(A.astype(np.float32))
                    # σ_g is already embedded in f_i^adv ≈ -σ_g⟨E,∇L⟩; do NOT multiply again
                    param.data.add_(update.to(param.device), alpha=coeff)


# ─── client ─────────────────────────────────��───────────────────────────────��


class MOTAR_Client(pFL_Client):

    def __init__(self, configs, id: int, times: int) -> None:
        super().__init__(configs, id, times)
        # Per-client personalized adapter Δ_i — lives on client, never communicated
        self.delta: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param, device="cpu")
            for name, param in self.model.named_parameters()
        }
        self._fitness: float = 0.0
        self._A_factors: Dict[str, np.ndarray] = {}
        self._B_factors: Dict[str, Optional[np.ndarray]] = {}

    def _offload_package(
        self, package: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """CPU-ify the standard entries plus MOTAR's ``delta`` tensors for return.

        Called only on the Ray worker path so that GPU tensors cross the process
        boundary safely.  The base implementation handles ``model`` and
        ``optimizer_state``; this override adds ``delta``.

        Parameters
        ----------
        package : dict or None
            The dict returned by :meth:`train`.

        Returns
        -------
        dict or None
            The same mapping with GPU tensors moved to CPU.
        """
        package = super()._offload_package(package)
        if package is not None and "delta" in package:
            package["delta"] = {k: v.cpu().clone() for k, v in package["delta"].items()}
        return package

    def variables_to_be_sent(self) -> Dict[str, Any]:
        """Send advantage fitness + low-rank factors. No seed. No model."""
        return {
            "fitness": self._fitness,
            "A_factors": self._A_factors,
            "B_factors": self._B_factors,
            "score": self.train_samples,
        }

    def _apply_delta(self, add: bool) -> None:
        sign = 1 if add else -1
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                d = self.delta.get(name)
                if d is not None:
                    param.data.add_(d.to(param.device), alpha=sign)

    def get_train_loss(self) -> float:
        """Personalized train loss: L_i(M + Δ_i)."""
        self._apply_delta(add=True)
        result = super().get_train_loss()
        self._apply_delta(add=False)
        return result

    def get_test_loss(self) -> float:
        """Personalized test loss: L_i(M + Δ_i)."""
        self._apply_delta(add=True)
        result = super().get_test_loss()
        self._apply_delta(add=False)
        return result

    def train(self) -> Dict[str, Any]:
        seed = self._loader_seed("train")
        self._set_worker_seed(seed)
        start_time = time.time()

        device = self.device
        train_loader = self.load_train_data()
        batch_iter = iter(train_loader)

        def next_batch():
            nonlocal batch_iter
            try:
                return next(batch_iter)
            except StopIteration:
                batch_iter = iter(self.load_train_data())
                return next(batch_iter)

        # ── Step A: K local ES steps → update Δ_i (stays on client, not uploaded) ──
        for _ in range(self.motar_K):
            batch = next_batch()
            A_loc, B_loc = _make_lora_factors(
                self.model.named_parameters(), self.motar_rank
            )
            # E_loc_eval: σ_l-scaled, for forward-pass perturbation only
            E_loc_eval = _lora_to_dict(
                A_loc,
                B_loc,
                self.model.named_parameters(),
                self.motar_rank,
                sigma=self.motar_sigma_l,
            )
            # E_loc_dir: pure direction (σ=1), for Δ_i update — σ_l already embedded in f_adv_local
            E_loc_dir = _lora_to_dict(
                A_loc, B_loc, self.model.named_parameters(), self.motar_rank, sigma=1.0
            )
            b = _eval_loss(self.model, self.delta, None, batch, self.loss, device)
            f_pert = _eval_loss(
                self.model, self.delta, E_loc_eval, batch, self.loss, device
            )
            f_adv_local = b - f_pert

            # guard: skip step if fitness is degenerate (σ_l outside linearization regime)
            if not math.isfinite(f_adv_local) or abs(f_adv_local) > 1e6:
                continue

            step = (self.motar_alpha_l / self.motar_K) * f_adv_local
            with torch.no_grad():
                for name, d in self.delta.items():
                    if name in E_loc_dir:
                        d.add_(torch.from_numpy(E_loc_dir[name]), alpha=step)

        # ── Step B: global perturbation — same batch for baseline + perturbed ──────
        A_glob, B_glob = _make_lora_factors(
            self.model.named_parameters(), self.motar_rank
        )
        # E_glob_eval: σ_g-scaled, for forward-pass only; A/B factors sent to server
        E_glob_eval = _lora_to_dict(
            A_glob,
            B_glob,
            self.model.named_parameters(),
            self.motar_rank,
            sigma=self.motar_sigma_g,
        )
        batch = next_batch()
        b_global = _eval_loss(self.model, self.delta, None, batch, self.loss, device)
        f_pert_global = _eval_loss(
            self.model, self.delta, E_glob_eval, batch, self.loss, device
        )

        fitness_raw = b_global - f_pert_global
        # guard: zero out degenerate global fitness so server update is a no-op
        self._fitness = (
            fitness_raw
            if math.isfinite(fitness_raw) and abs(fitness_raw) <= 1e6
            else 0.0
        )
        self._A_factors = A_glob
        self._B_factors = B_glob
        self.metrics["train_loss"].append(b_global)

        if self.efficiency in ("low", "med"):
            self.model.to("cpu")

        return {
            "model": self.model,
            "optimizer_state": self.optimizer,
            "train_time": time.time() - start_time,
            "train_samples": self.train_samples,
            "fitness": self._fitness,
            "A_factors": self._A_factors,
            "B_factors": self._B_factors,
            "delta": self.delta,
        }
