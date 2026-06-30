from collections import OrderedDict
from typing import Any, Dict, Optional

import torch

from .FedRidge import FedRidge, FedRidge_Client


class DLSA(FedRidge):
    """DLSA: Distributed Least Squares Approximation (Zhu et al. JCGS 2021).

    Applies the Gaussian-NLL specialisation of the paper's WLSE (Eq. 2.4) to
    federated LTSF.  Each client uploads its local OLS solution W_i and the
    normalised precision matrix Sigma_k^{-1} = Sigma_xx_i / (n_i * sigma_i^2).
    The server aggregates via:

        W_g = (Σ_k α_k Σ_k^{-1})^{-1} (Σ_k α_k Σ_k^{-1} W_k)
            where α_k = n_k / N

    No personalisation — global model only.

    References
    ----------
    Zhu et al., "Least-Square Approximation for a Distributed System,"
    JCGS 2021. https://doi.org/10.1080/10618600.2021.1923517
    """

    optional = {"gamma": 0.0}

    def aggregate_client_updates(self, packages) -> None:
        L = self.input_len
        H = self.output_len

        N = sum(packages[cid]["n_i"] for cid in packages)
        h_sum = torch.zeros(L, L)
        hw_sum = torch.zeros(L, H)
        for cid in packages:
            alpha_k = packages[cid]["n_i"] / N
            h_k = packages[cid]["h_i"]  # = Sigma_xx / (n_i * sigma_i^2) = Sigma_k^{-1}
            h_sum.add_(h_k, alpha=alpha_k)
            hw_sum.add_(h_k @ packages[cid]["w_i"], alpha=alpha_k)

        W_g = torch.linalg.solve(h_sum + self.gamma * torch.eye(L), hw_sum)
        self._load_linear_weights(self.model, W_g)
        self._commit_global(
            OrderedDict(
                (k, v.detach().cpu().clone()) for k, v in self.model.named_parameters()
            )
        )


class DLSA_Client(FedRidge_Client):
    """Client for DLSA.

    Computes local OLS solution W_i and normalised precision matrix
    Sigma_k^{-1} = Sigma_xx_i / (n_i * sigma_i^2) per paper Eq. 2.4.
    """

    _w_i: Optional[torch.Tensor] = None
    _h_i: Optional[torch.Tensor] = None
    _n_i: int = 0

    def fit(self) -> None:
        self._set_worker_seed(self._loader_seed("train"))
        loader = self.load_train_data()
        L = self.input_len
        H = self.output_len

        sigma_xx = torch.zeros(L, L)
        sigma_xy = torch.zeros(L, H)
        yy_sum = 0.0
        n_obs = 0

        for batch_x, batch_y, *_ in loader:
            B, _, C = batch_x.shape
            x = batch_x.permute(0, 2, 1).reshape(B * C, L)
            y = batch_y.permute(0, 2, 1).reshape(B * C, H)
            sigma_xx.add_(x.T @ x)
            sigma_xy.add_(x.T @ y)
            yy_sum += (y * y).sum().item()
            n_obs += B * C

        sxx = sigma_xx / n_obs
        sxy = sigma_xy / n_obs

        w_i = torch.linalg.solve(sxx, sxy)

        yy_mean = yy_sum / n_obs
        fit_trace = (w_i * sxy).sum().item()
        sigma_sq = max((yy_mean - fit_trace) / H, 1e-8)

        self._w_i = w_i
        self._h_i = sxx / sigma_sq  # = Sigma_xx / (n_i * sigma_i^2) = Sigma_k^{-1}
        self._n_i = n_obs
        self.train_samples = n_obs

    def package(self) -> Dict[str, Any]:
        result = super().package()
        result["w_i"] = self._w_i
        result["h_i"] = self._h_i
        result["n_i"] = self._n_i
        result["__wire__"] = ("w_i", "h_i", "n_i")
        return result
