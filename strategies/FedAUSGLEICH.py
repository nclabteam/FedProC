# -*- coding: utf-8 -*-
"""
FedAUSGLEICH — Adaptive MP-precision one-shot analytic federated learning.

Corrected adaptive regularisation (v13 formulas):

  λ_v13 = min(HarmonicMean(σ²_MP_i) × q_pooled, 0.01)
    where q_pooled = L / Σ n_windows_i  (uploaded as 1 extra integer per client)
    Introduces missing sample-size dependence: more total data → smaller λ.

  γ_v13 = noise_frac²_i × cos_sim(r_i, r_g)
    noise_frac_i = σ²_MP_i / r_i[0]
    Quadratic squaring corrects the scale: (0.1)² = 0.01 ≈ sweep-optimal
    for ETH/Elec (linear gave 0.1, 10× too large).

Communication cost: O(L+H+1) floats + 1 integer per client per direction.
Personalization is computed server-side using uploaded Sigma_xy.

See: D:\\01.Code\\neko-vault\\proposed\\FedTELOS\\reports\\notes-adaptive-hyperparams.md
"""

import time
from collections import OrderedDict
from typing import Any, Dict, Optional

import ray
import torch

from .FedRidge import _LinearWeightsMixin
from .pFL import pFL, pFL_Client


class _ToeplitzMixin:
    """Shared mixin: Toeplitz/Hankel reconstruction and autocorrelation extraction."""

    @staticmethod
    def _toeplitz(r: torch.Tensor) -> torch.Tensor:
        L = r.shape[0]
        idx = torch.arange(L)
        i, j = torch.meshgrid(idx, idx, indexing="ij")
        return r[torch.abs(i - j)]

    @staticmethod
    def _hankel(r_i: torch.Tensor, s_i: torch.Tensor) -> torch.Tensor:
        L = r_i.shape[0]
        H = s_i.shape[0]
        full_r = torch.cat([r_i, s_i])
        j_idx = torch.arange(L, device=r_i.device).unsqueeze(1)
        h_idx = torch.arange(H, device=r_i.device).unsqueeze(0)
        return full_r[L + h_idx - j_idx]

    @staticmethod
    def _extract_autocorr(Sigma: torch.Tensor) -> torch.Tensor:
        L = Sigma.shape[0]
        r = torch.empty(L)
        for k in range(L):
            r[k] = Sigma.diagonal(offset=k).mean()
        return r


class FedAUSGLEICH(_ToeplitzMixin, _LinearWeightsMixin, pFL):
    """
    FedAUSGLEICH server — corrected adaptive MP-precision one-shot analytic FL.

    λ = min(HarmonicMean(σ²_MP_i) × q_pooled, 0.01)
      where q_pooled = L / Σ n_windows_i

    γ_i = noise_frac²_i × cos_sim(r_i, r_g)  [computed server-side]

    Personalization is computed server-side in aggregate_client_updates using
    the Sigma_xy uploaded by each client; results are written directly to
    clients_personal_model_params for evaluation.
    """

    optional = {}  # λ and γ are adaptive — no tunable hyperparameters

    def train(self) -> None:
        self.logger.info(
            "%s: one-shot sufficient-statistics FL (v13 adaptive λ/γ, q-normalised)",
            self.__class__.__name__,
        )
        round_start = time.time()
        self.current_iter = 0
        self.selected_clients = [i for i in range(self.num_clients) if not self.is_new[i]]
        packages = self.trainer.train(self.selected_clients)
        uplink, downlink = self._compute_send_mb(packages)
        self.metrics["downlink_mb"].append(downlink)
        for cid, mb in uplink.items():
            self._ensure_client_row(cid)["uplink_mb"][-1] = mb
        self.aggregate_client_updates(packages)

        for dataset_type in ["train", "test"]:
            if dataset_type == "train" and self.skip_eval_train:
                continue
            if not self.exclude_server_model_processes:
                self.evaluate_generalization(dataset_type)
            self._pre_eval_hook(dataset_type)

        self.metrics["time_per_iter"].append(time.time() - round_start)
        self.fix_results(default=self.default_value)
        self.save_results()
        self._save_per_client_results()
        try:
            self.close_logger()
        except Exception:
            pass
        try:
            ray.shutdown()
        except Exception:
            pass

    def aggregate_client_updates(self, packages) -> None:
        L = self.input_len
        H = self.output_len

        # --- precision-weighted global autocorrelation vectors ---
        r_prec = torch.zeros(L)
        s_prec = torch.zeros(H)
        prec_total = 0.0
        n_win_total = 0

        for pkg in packages.values():
            prec = 1.0 / pkg["sigma_sq"]
            r_prec.add_(pkg["r_i"], alpha=prec)
            s_prec.add_(pkg["s_i"], alpha=prec)
            prec_total += prec
            n_win_total += int(pkg["n_windows"])

        r_g = r_prec / prec_total
        s_g = s_prec / prec_total

        # --- adaptive λ (v13) ---
        N = len(packages)
        sigma_sq_g = N / prec_total  # = HarmonicMean(σ²_MP_i)
        q_pooled = L / n_win_total if n_win_total > 0 else 1.0
        lam = min(sigma_sq_g * q_pooled, 0.01)

        # --- per-client personalization (server-side, using uploaded Sigma_xy) ---
        norm_rg = torch.norm(r_g).item()
        gamma_list = []

        for cid, pkg in packages.items():
            nf = pkg["sigma_sq"] / (pkg["r_i"][0].item() + 1e-8)
            norm_ri = torch.norm(pkg["r_i"]).item()
            sim = torch.dot(pkg["r_i"], r_g).item() / (norm_ri * norm_rg + 1e-8)
            sim = max(sim, 0.0)
            gamma = nf ** 2 * sim
            gamma_list.append(gamma)

            A_i = (
                self._toeplitz(pkg["r_i"])
                + gamma * self._toeplitz(r_g)
                + gamma * torch.eye(L)
            )
            b_i = pkg["Sigma_xy"] + gamma * self._hankel(r_g, s_g)
            W_i = torch.linalg.solve(A_i, b_i)
            self._load_linear_weights(self.model, W_i)
            self.clients_personal_model_params[cid].update(
                {k: v.detach().cpu().clone() for k, v in self.model.named_parameters()}
            )

        g_mean = sum(gamma_list) / len(gamma_list)
        self.logger.info(
            "%s: λ=%.6f (σ²_g=%.4f q_pooled=%.3e N=%d) | γ mean=%.4f",
            self.__class__.__name__,
            lam,
            sigma_sq_g,
            q_pooled,
            N,
            g_mean,
        )
        self.metrics.setdefault("lambda_auto", []).append(lam)
        self.metrics.setdefault("sigma_sq_g", []).append(sigma_sq_g)
        self.metrics.setdefault("q_pooled", []).append(q_pooled)
        self.metrics.setdefault("gamma_auto_mean", []).append(g_mean)

        # --- global model ---
        W_g = torch.linalg.solve(
            self._toeplitz(r_g) + lam * torch.eye(L),
            self._hankel(r_g, s_g),
        )
        self._load_linear_weights(self.model, W_g)
        self._commit_global(
            OrderedDict(
                (k, v.detach().cpu().clone()) for k, v in self.model.named_parameters()
            )
        )


class FedAUSGLEICH_Client(_ToeplitzMixin, _LinearWeightsMixin, pFL_Client):
    """
    FedAUSGLEICH client.

    Upload: (r_i, s_i, Sigma_xy, σ²_MP_i, n_windows_i) — L+H+L*H+1+1 values.
    Personalization is computed server-side; clients_personal_model_params is
    written directly by the server's aggregate_client_updates.
    """

    _r_i: Optional[torch.Tensor] = None
    _s_i: Optional[torch.Tensor] = None
    _Sigma_xy: Optional[torch.Tensor] = None
    _sigma_sq: float = 1.0
    _n_windows: int = 0

    def fit(self) -> None:
        self._set_worker_seed(self._loader_seed("train"))
        loader = self.load_train_data()
        L = self.input_len
        H = self.output_len

        Sigma_xx = torch.zeros(L, L)
        Sigma_xy = torch.zeros(L, H)
        n_obs = 0
        n_windows = 0

        for batch_x, batch_y, *_ in loader:
            B, _, C = batch_x.shape
            x = batch_x.permute(0, 2, 1).reshape(B * C, L)
            y = batch_y.permute(0, 2, 1).reshape(B * C, H)
            Sigma_xx.add_(x.T @ x)
            Sigma_xy.add_(x.T @ y)
            n_obs += B * C
            n_windows += B

        Sxx = Sigma_xx / n_obs
        Sxy = Sigma_xy / n_obs

        self._r_i = self._extract_autocorr(Sxx)
        self._s_i = Sxy[0, :]
        self._Sigma_xy = Sxy
        self._n_windows = n_windows

        eigs = torch.linalg.eigvalsh(Sxx)
        q = L / max(n_windows, 1)
        sigma_sq_0 = eigs.mean().item()
        lambda_plus = sigma_sq_0 * (1.0 + q ** 0.5) ** 2
        noise_eigs = eigs[eigs <= lambda_plus]
        self._sigma_sq = max(
            noise_eigs.mean().item() if noise_eigs.numel() > 0 else sigma_sq_0,
            1e-8,
        )


    def package(self, train_time: float) -> Dict[str, Any]:
        result = super().package(train_time)
        result["r_i"] = self._r_i
        result["s_i"] = self._s_i
        result["Sigma_xy"] = self._Sigma_xy
        result["sigma_sq"] = self._sigma_sq
        result["n_windows"] = self._n_windows
        return result
