from collections import OrderedDict
from typing import Any, Dict, Optional

import torch

from .FedRidge import FedRidge, FedRidge_Client


class FedDLSA(FedRidge):
    """FedDLSA: Distributed Least Squares Approximation (Zhu et al. 2021).

    Applies the Gaussian-NLL specialisation to federated LTSF.  Each client
    uploads its local OLS solution W_i and a precision matrix H_i =
    Sigma_xx_i / sigma_i^2, where sigma_i^2 is the client's estimated noise
    variance.  The server aggregates via the WLSE:

        W_g = (Σ_i H_i + γI)^{-1} Σ_i H_i W_i

    This down-weights noisy clients, reducing to standard FedRidge (γ=0) when
    all sigma_i^2 are equal.

    No personalisation — global model only.

    References
    ----------
    Jun Zhu et al., "Least-Square Approximation for a Distributed System,"
    JCGS 2021. https://doi.org/10.1080/10618600.2021.1923517
    """

    optional = {"gamma": 0.0}

    def aggregate_client_updates(self, packages) -> None:
        L = self.input_len
        H = self.output_len
        cids = list(packages.keys())

        h_sum = torch.zeros(L, L)
        hw_sum = torch.zeros(L, H)
        for cid in cids:
            h_sum.add_(packages[cid]["h_i"])
            hw_sum.add_(packages[cid]["h_i"] @ packages[cid]["w_i"])

        W_g = torch.linalg.solve(h_sum + self.gamma * torch.eye(L), hw_sum)
        self._load_linear_weights(self.model, W_g)
        self._commit_global(
            OrderedDict(
                (k, v.detach().cpu().clone()) for k, v in self.model.named_parameters()
            )
        )


class FedDLSA_Client(FedRidge_Client):
    """Client for FedDLSA.

    Computes local OLS solution W_i and heteroscedasticity-aware precision
    matrix H_i = Sigma_xx_i / sigma_i^2 in a single data pass, where
    sigma_i^2 is estimated from the training residuals.

    No personalisation — receives and loads the global model only.
    """

    _w_i: Optional[torch.Tensor] = None
    _h_i: Optional[torch.Tensor] = None

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
        self._h_i = sigma_xx / sigma_sq
        self.train_samples = n_obs

    def package(self) -> Dict[str, Any]:
        result = super().package()
        result["w_i"] = self._w_i
        result["h_i"] = self._h_i
        return result
