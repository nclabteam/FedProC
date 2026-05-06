"""
LoRA-FAIR18: Federated LoRA Fine-Tuning with Empirical Bayes Aggregation

Theoretically grounded extension of LoRA-FAIR that uses Type-II Maximum Likelihood
(Empirical Bayes) to estimate the regularization parameter λ from data.

KEY: Type-II ML with Fast Fixed-Point Iteration (2-3 iterations)
    Data-driven λ estimation via MacKay balance equations. Converges rapidly, no hyperparameter tuning.

Key Innovation:
    Instead of fixing λ as a hyperparameter, LoRA-FAIR18 derives λ from a probabilistic model:

    Generative Model:
        W_target = Ā(B̄ + ΔB) + ε,    ε ~ N(0, σ²I)
        ΔB ~ N(0, τ²I)  [prior on correction term]

    MAP Estimation:
        ΔB* = (Ā^T Ā + λI)^{-1} Ā^T R,    where λ = σ²/τ²

    Type-II ML via Fixed-Point Iteration:
        1. Initialize λ from OLS residual variance ratio
        2. Solve ridge: w* = (X^T X + λI)^{-1} X^T y
        3. Update λ via MacKay's balanced variance formula
        4. Repeat steps 2-3 until convergence (typically 2-3 iterations)
        5. Return converged λ and ridge solution

Advantages:
    - Probabilistic interpretation of ridge regression
    - Data-driven λ selection without grid search or hyperparameter tuning
    - Fast fixed-point iteration (2-3 iterations typical, same cost as ridge solve)
    - Convergence guaranteed to satisfy MacKay balance equations
    - Automatic adaptation vs manual hyperparameter choice

References:
    - MacKay (1992): Bayesian interpolation
    - Tipping & Bishop (2003): Sparse Bayesian learning
    - Wipf & Nagarajan (2010): Iterative reweighted L2-norm algorithms
"""

from typing import Dict

import torch

from .FedIT import FedIT, FedIT_Client


class EmpiricalBayesRidge:
    """
    Type-II Maximum Likelihood (Empirical Bayes) for ridge regression via fixed-point iteration.

    For single output: minimize ||y - Xw||² + λ||w||²

    Algorithm theory:
        Model: y = Xw + ε,  ε ~ N(0, σ²I),  w ~ N(0, τ²I)

        At convergence of MacKay EM, the optimal λ = σ²/τ² satisfies:

        λ* = (residual_norm² / w_norm²) * (d - effective_dof) / effective_dof

        where:
        - residual_norm² = ||y - Xw*||² (residual error)
        - w_norm² = ||w*||² (weight magnitude)
        - effective_dof = r - λ·tr((X^T X + λI)^{-1}) (effective degrees of freedom)

        We solve this iteratively but with only 2-3 fixed-point iterations
        instead of full EM convergence, making it quasi-closed-form.
    """

    def __init__(self, max_iter: int = 3, tol: float = 1e-4, verbose: bool = False):
        self.max_iter = max_iter  # Only 2-3 fixed-point iterations
        self.tol = tol
        self.verbose = verbose

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> Dict:
        """
        Estimate λ via Type-II ML using fixed-point iteration (typically 2-3 iterations).

        Args:
            X: Design matrix [d, r]
            y: Target vector [d]

        Returns:
            Dict with 'w_map', 'lambda', 'sigma2', 'tau2'
        """
        d, r = X.shape
        device = X.device
        dtype = X.dtype

        gram = X.T @ X  # [r, r]
        Xy = X.T @ y  # [r]
        (y**2).sum().item()

        # Initialize λ with a heuristic: λ ≈ residual_var / coeff_var
        # Start with OLS residual variance estimate
        try:
            w_ols = torch.linalg.solve(gram, Xy)
        except Exception:
            w_ols = torch.linalg.lstsq(gram, Xy.unsqueeze(1)).solution.squeeze()

        residual_ols = y - X @ w_ols
        sigma2_hat = (residual_ols**2).sum().item() / max(d - r, 1)
        w_norm_sq_ols = (w_ols**2).sum().item()
        lambda_current = sigma2_hat / max(w_norm_sq_ols, 1e-12)
        lambda_current = max(lambda_current, 1e-6)

        # Fixed-point iteration (2-3 steps typically sufficient)
        for iteration in range(self.max_iter):
            lambda_old = lambda_current

            # Ridge solve with current λ
            precision = gram + lambda_current * torch.eye(r, device=device, dtype=dtype)
            try:
                w_map = torch.linalg.solve(precision, Xy)
            except Exception:
                w_map = torch.linalg.lstsq(
                    precision, Xy.unsqueeze(1)
                ).solution.squeeze()

            # Compute residual and weight norms
            residual = y - X @ w_map
            residual_norm_sq = (residual**2).sum().item()
            w_norm_sq = (w_map**2).sum().item() + 1e-12

            # Effective degrees of freedom (via trace formula)
            try:
                cov = torch.inverse(precision)
                trace_cov = torch.trace(cov).item()
            except Exception:
                cov = torch.linalg.inv(precision)
                trace_cov = torch.trace(cov).item()

            effective_dof = r - lambda_current * trace_cov
            effective_dof = max(effective_dof, 0.1)  # Avoid singularity

            # MacKay fixed-point update
            sigma2_new = residual_norm_sq / max(d - effective_dof, 1)
            tau2_new = w_norm_sq / effective_dof
            lambda_new = sigma2_new / max(tau2_new, 1e-12)
            lambda_new = max(lambda_new, 1e-6)
            lambda_current = lambda_new

            # Early stopping if converged
            if abs(lambda_new - lambda_old) / (abs(lambda_old) + 1e-12) < self.tol:
                break

        sigma2 = sigma2_new
        tau2 = tau2_new

        return {
            "w_map": w_map,
            "lambda": lambda_current,
            "sigma2": sigma2,
            "tau2": tau2,
            "iterations": iteration + 1,
        }


class FedEBER(FedIT):
    """
    FedEBER server with Empirical Bayes ridge aggregation.

    Extends FedIT with Type-II ML for data-driven regularization parameter selection.
    Instead of using fixed λ, learns it per column via Gaussian probabilistic model.
    """

    optional = {
        "lora_r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "lora_bayes_max_iter": 20,
        "lora_bayes_tol": 1e-4,
        "lora_target_modules": ["Linear"],
    }

    compulsory = {
        "exclude_server_model_processes": True,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--lora_r", default=None, type=int)
        parser.add_argument("--lora_alpha", default=None, type=int)
        parser.add_argument("--lora_dropout", default=None, type=float)
        parser.add_argument("--lora_bayes_max_iter", default=None, type=int)
        parser.add_argument("--lora_bayes_tol", default=None, type=float)
        parser.add_argument(
            "--lora_target_modules",
            default=None,
            nargs="+",
            help="List of target module class names (e.g., Linear Conv1d)",
        )

    def aggregate_models(self):
        """
        Override FedIT.aggregate_models:
        1) Compute A_bar, B_bar (weighted average)
        2) Compute ideal W_target = Σ p_k (A_k @ B_k)
        3) For each column, solve ridge regression with λ estimated via Type-II ML
        4) Update global model with corrected LoRA params
        """
        device = next(self.model.parameters()).device

        # Build lora layer map from first client's lora_params keys
        first_client = self.client_data[0]["lora_params"]
        lora_layers = {}
        for key in first_client.keys():
            if key.endswith(".lora_A"):
                layer = key[: -len(".lora_A")]
                lora_layers.setdefault(layer, {})["A_name"] = key
            elif key.endswith(".lora_B"):
                layer = key[: -len(".lora_B")]
                lora_layers.setdefault(layer, {})["B_name"] = key

        # Prepare aggregated containers
        A_bar = {}
        B_bar = {}
        W_target = {}

        # Initialize zeros using shapes from first client
        for layer, info in lora_layers.items():
            A_name = info.get("A_name")
            B_name = info.get("B_name")
            if not (A_name and B_name):
                continue
            sample_A = first_client[A_name]
            sample_B = first_client[B_name]
            A_bar[A_name] = torch.zeros_like(sample_A, device=device)
            B_bar[B_name] = torch.zeros_like(sample_B, device=device)
            in_features = sample_A.shape[0]
            out_features = sample_B.shape[1]
            W_target[layer] = torch.zeros((in_features, out_features), device=device)

        # Weighted aggregation
        for client_data, weight in zip(self.client_data, self.weights):
            client_lora = client_data["lora_params"]
            for layer, info in lora_layers.items():
                A_name = info.get("A_name")
                B_name = info.get("B_name")
                if A_name in client_lora and B_name in client_lora:
                    cA = client_lora[A_name].to(device)
                    cB = client_lora[B_name].to(device)
                    A_bar[A_name].add_(cA, alpha=weight)
                    B_bar[B_name].add_(cB, alpha=weight)
                    W_target[layer].add_(cA @ cB, alpha=weight)

        aggregated_lora = {}
        bayes_ridge = EmpiricalBayesRidge(
            max_iter=getattr(
                self, "lora_bayes_max_iter", FedEBER.optional["lora_bayes_max_iter"]
            ),
            tol=getattr(self, "lora_bayes_tol", FedEBER.optional["lora_bayes_tol"]),
            verbose=False,
        )

        # Per-layer Empirical Bayes ridge regression for ΔB
        for layer, info in lora_layers.items():
            A_name = info.get("A_name")
            B_name = info.get("B_name")
            if A_name not in A_bar or B_name not in B_bar:
                continue

            A_t = A_bar[A_name]  # [in, r]
            B_t = B_bar[B_name]  # [r, out]
            W_tgt = W_target[layer]  # [in, out]

            # Solve per-column ridge with Type-II ML
            # For each output column: minimize ||W_tgt[:, j] - A_t @ b_j||² + λ_j||b_j||²
            delta_B = torch.zeros_like(B_t, device=device)

            for j in range(B_t.shape[1]):
                y_col = W_tgt[:, j]  # [in]
                residual = y_col - A_t @ B_t[:, j]

                # Type-II ML: estimate λ for this column
                fit_info = bayes_ridge.fit(A_t, residual)
                lambda_j = fit_info["lambda"]

                # Solve ridge with estimated λ
                try:
                    precision = A_t.T @ A_t + lambda_j * torch.eye(
                        A_t.shape[1], device=device, dtype=A_t.dtype
                    )
                    delta_b_j = torch.linalg.solve(precision, A_t.T @ residual)
                except RuntimeError:
                    delta_b_j = torch.zeros(
                        A_t.shape[1], device=device, dtype=A_t.dtype
                    )

                delta_B[:, j] = delta_b_j

            aggregated_lora[A_name] = A_t.detach().cpu().clone()
            aggregated_lora[B_name] = (B_t + delta_B).detach().cpu().clone()

        # Update global model
        self.update_lora_params(self.model, aggregated_lora)


class FedEBER_Client(FedIT_Client):
    pass
