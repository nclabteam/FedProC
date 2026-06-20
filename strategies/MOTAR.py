"""MOTAR: Advantage-fitness Federated Evolution Strategies.

Each FL client = one ES population member. Client generates a random low-rank
perturbation (A_i, B_i), computes advantage fitness
  f_i^adv = L_i(M + Δ_i; B) - L_i(M + Δ_i + σ_g·E_i; B)
then uploads (f_i^adv, A_i, B_i) to the server.

Server performs the EGGROLL update:
  M ← M + α_g · Σ_i  k_i · f_i^adv · (1/√r) A_i B_i^T
"""

import math
from collections import OrderedDict
from typing import Dict, Optional

import numpy as np
import torch

from .pFL import pFL, pFL_Client
from .tFL import tFL


def _make_lora_factors(
    named_params, rank: int
) -> tuple:
    """Sample random A, B Gaussian matrices for each parameter tensor.

    For 1-D params (bias), A carries the full perturbation and B is None.
    Returns two dicts {name: np.ndarray}.
    """
    A_factors: Dict[str, np.ndarray] = {}
    B_factors: Dict[str, Optional[np.ndarray]] = {}
    for name, param in named_params:
        shape = tuple(param.shape)
        total = int(np.prod(shape))
        if len(shape) >= 2:
            m = total // shape[-1]
            n = shape[-1]
            A = np.random.randn(m, rank).astype(np.float32)
            B = np.random.randn(rank, n).astype(np.float32)
        else:
            A = np.random.randn(total, 1).astype(np.float32)
            B = None
        A_factors[name] = A
        B_factors[name] = B
    return A_factors, B_factors


def _lora_to_dict(
    A_factors: Dict[str, np.ndarray],
    B_factors: Dict[str, Optional[np.ndarray]],
    named_params,
    rank: int,
    sigma: float,
) -> Dict[str, np.ndarray]:
    """Reconstruct the perturbation E = σ/√r · A B^T for each parameter."""
    result: Dict[str, np.ndarray] = {}
    for name, param in named_params:
        A = A_factors.get(name)
        if A is None:
            continue
        B = B_factors.get(name)
        shape = tuple(param.shape)
        if B is not None:
            E = (A @ B.T) / np.sqrt(rank)
        else:
            E = A[:, 0]
        result[name] = (sigma * E).reshape(shape).astype(np.float32)
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


# ─── server ────────────────────────────────────────────────────────────────


class MOTAR(pFL):

    optional = {
        "motar_rank": 4,
        "motar_sigma_g": 1e-4,
        "motar_sigma_l": 1e-3,
        "motar_K": 3,
        "motar_alpha_g": 1.0,
        "motar_alpha_l": 1.0,
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

    def aggregate_client_updates(self, packages) -> None:
        """EGGROLL update: M ← M + α_g · Σ k_i · f_i^adv · (1/√r) A_i B_i^T"""
        scores = [p["score"] for p in packages.values()]
        total = float(sum(scores))
        cids = list(packages.keys())

        new_params = OrderedDict(
            (name, tensor.clone()) for name, tensor in self.public_model_params.items()
        )

        with torch.no_grad():
            for cid, weight in zip(cids, [s / total for s in scores]):
                pkg = packages[cid]
                fitness = float(pkg.get("fitness", 0.0))
                A_factors = pkg.get("A_factors", {})
                B_factors = pkg.get("B_factors", {})
                coeff = self.motar_alpha_g * weight * fitness

                for name in new_params:
                    A = A_factors.get(name)
                    if A is None:
                        continue
                    B = B_factors.get(name)
                    shape = tuple(new_params[name].shape)
                    if B is not None:
                        E = (A @ B.T) / np.sqrt(self.motar_rank)
                        update = torch.from_numpy(E.reshape(shape))
                    else:
                        update = torch.from_numpy(A.astype(np.float32))
                    new_params[name] = new_params[name] + update.to(new_params[name].device) * coeff

        self._commit_global(new_params)


# ─── client ────────────────────────────────────────────────────────────────


class MOTAR_Client(pFL_Client):

    def __init__(self, configs, times, device) -> None:
        super().__init__(configs=configs, times=times, device=device)
        self.delta: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param, device="cpu")
            for name, param in self.model.named_parameters()
        }
        self._fitness: float = 0.0
        self._A_factors: Dict[str, np.ndarray] = {}
        self._B_factors: Dict[str, Optional[np.ndarray]] = {}

    def set_parameters(self, package: dict) -> None:
        super().set_parameters(package)
        personal = package["personal_model_params"]
        if personal:
            delta_keys = {
                k[len("__delta_"):]: personal[k]
                for k in personal
                if k.startswith("__delta_")
            }
            for name in self.delta:
                if name in delta_keys:
                    self.delta[name] = delta_keys[name]

    def fit(self) -> None:
        seed = self._loader_seed("train")
        self._set_worker_seed(seed)

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

        # ── Step A: K local ES steps → update Δ_i ──
        for _ in range(self.motar_K):
            batch = next_batch()
            A_loc, B_loc = _make_lora_factors(
                self.model.named_parameters(), self.motar_rank
            )
            E_loc_eval = _lora_to_dict(
                A_loc,
                B_loc,
                self.model.named_parameters(),
                self.motar_rank,
                sigma=self.motar_sigma_l,
            )
            E_loc_dir = _lora_to_dict(
                A_loc, B_loc, self.model.named_parameters(), self.motar_rank, sigma=1.0
            )
            b = _eval_loss(self.model, self.delta, None, batch, self.loss, device)
            f_pert = _eval_loss(
                self.model, self.delta, E_loc_eval, batch, self.loss, device
            )
            f_adv_local = b - f_pert

            if not math.isfinite(f_adv_local) or abs(f_adv_local) > 1e6:
                continue

            step = (self.motar_alpha_l / self.motar_K) * f_adv_local
            with torch.no_grad():
                for name, d in self.delta.items():
                    if name in E_loc_dir:
                        d.add_(torch.from_numpy(E_loc_dir[name]), alpha=step)

        # ── Step B: global perturbation ──
        A_glob, B_glob = _make_lora_factors(
            self.model.named_parameters(), self.motar_rank
        )
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

    def package(self, train_time: float) -> dict:
        result = super().package(train_time)
        result["fitness"] = self._fitness
        result["A_factors"] = self._A_factors
        result["B_factors"] = self._B_factors
        # Persist delta in personal_model_params
        for name, d in self.delta.items():
            result["personal_model_params"][f"__delta_{name}"] = d.cpu().clone()
        return result

    def evaluate_personalized(
        self,
        client_id: int,
        global_params,
        personal_params,
        dataset_type: str,
        current_iter: int,
    ) -> float:
        self.id = client_id
        self.current_iter = current_iter
        self._load_private(client_id)
        self.model.load_state_dict(global_params, strict=False)
        # Apply stored delta
        if personal_params:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    key = f"__delta_{name}"
                    if key in personal_params:
                        param.data.add_(personal_params[key].to(param.device))
        loader = (
            self.load_test_data()
            if dataset_type == "test"
            else self.load_train_data()
        )
        losses = self.calculate_loss(
            model=self.model,
            dataloader=loader,
            criterion=self.loss,
            device=self.device,
            offload_after=self.efficiency != "high",
        )
        return float(np.mean(losses))
