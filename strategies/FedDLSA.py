from typing import Any, Dict, Optional

import torch

from .FedRidge import FedRidge, FedRidge_Client


class FedDLSA(FedRidge):
    """
    FedDLSA: Distributed Least Squares Approximation (Zhu et al. 2021)
    applied to federated LTSF, using the Gaussian-NLL specialisation.

    The paper's WLSE for a general loss uses H_k = Omega_k (expected Hessian) as
    precision weight.  For the Gaussian NLL loss L = (y - x^T w)^2 / (2 sigma_k^2),
    the Hessian is X^T X / sigma_k^2, so H_i = Sigma_xx_i / sigma_i^2.  This
    down-weights noisy clients and reduces to global OLS (= FedRidge gamma=0) when
    all sigma_i^2 are equal.

    Using the plain squared loss (no sigma^2) would give H_i = Sigma_xx_i and the
    WLSE would collapse to FedRidge(gamma=0) regardless of client heteroscedasticity.

    Upload: W_i (local OLS, L×H) and H_i = Sigma_xx_i / sigma_i^2 (L×L, unnormalized).
    Server aggregates via precision-weighted combination:

        W_g = (sum_i H_i + gamma*I)^{-1} sum_i H_i @ W_i

    No personalization — global model only.

    Paper variants TWLSE and m-WLSE are not implemented: for the quadratic (squared)
    loss, Newton's method converges in one step, so TWLSE = WLSE for linear models.
    """

    optional = {"gamma": 0.0}

    def aggregate_models(self) -> None:
        L = self.input_len
        H = self.output_len
        gamma = getattr(self, "gamma", 0.0)

        H_sum = torch.zeros(L, L)
        HW_sum = torch.zeros(L, H)
        for cd in self.client_data:
            H_i = cd["H_i"]  # (L, L)
            W_i = cd["W_i"]  # (L, H)
            H_sum.add_(H_i)
            HW_sum.add_(H_i @ W_i)

        W_g = torch.linalg.solve(H_sum + gamma * torch.eye(L), HW_sum)
        self._load_linear_weights(self.model, W_g)

    def calculate_aggregation_weights(self) -> None:
        pass  # H_i encodes precision weighting; self.weights is unused


class FedDLSA_Client(FedRidge_Client):
    """
    Client for FedDLSA.

    Computes local OLS W_i and precision H_i = Sigma_xx_i / sigma_i^2 in one data pass.
    No personalization step — receives and loads global model only.
    """

    _W_i: Optional[torch.Tensor] = None
    _H_i: Optional[torch.Tensor] = None

    def compute_statistics(self) -> None:
        loader = self.load_train_data()
        L = self.input_len
        H = self.output_len

        Sigma_xx = torch.zeros(L, L)
        Sigma_xy = torch.zeros(L, H)
        yy_sum = 0.0
        n_obs = 0

        for batch_x, batch_y, *_ in loader:
            B, _, C = batch_x.shape
            x = batch_x.permute(0, 2, 1).reshape(B * C, L)
            y = batch_y.permute(0, 2, 1).reshape(B * C, H)
            Sigma_xx.add_(x.T @ x)
            Sigma_xy.add_(x.T @ y)
            yy_sum += (y * y).sum().item()
            n_obs += B * C

        Sxx = Sigma_xx / n_obs  # (L, L)
        Sxy = Sigma_xy / n_obs  # (L, H)

        # Local OLS: W_i = Sxx^{-1} Sxy
        W_i = torch.linalg.solve(Sxx, Sxy)

        # Residual variance: RSS/n_obs = trace(Syy) - trace(W_i^T Sxy)
        # Uses OLS optimality: trace(W_i^T Sxx W_i) = trace(W_i^T Sxy)
        yy_mean = yy_sum / n_obs
        fit_trace = (W_i * Sxy).sum().item()
        sigma_sq = max((yy_mean - fit_trace) / H, 1e-8)

        self._W_i = W_i
        self._H_i = Sigma_xx / sigma_sq

    def variables_to_be_sent(self) -> Dict[str, Any]:
        return {
            "W_i": self._W_i,
            "H_i": self._H_i,
        }

