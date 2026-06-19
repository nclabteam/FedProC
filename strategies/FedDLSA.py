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

    def _apply_client_result(
        self, client, package: Optional[Dict[str, Any]]
    ) -> None:
        """Apply local OLS and precision matrix returned by :meth:`FedDLSA_Client.train`.

        Parameters
        ----------
        client : FedDLSA_Client
            Original client object to update.
        package : dict or None
            Return value of ``FedDLSA_Client.train()``, containing ``w_i``,
            ``h_i``, and ``train_samples``.
        """
        if package is None:
            return
        client._w_i = package["w_i"]
        client._h_i = package["h_i"]
        client.train_samples = package["train_samples"]

    def aggregate_models(self) -> None:
        L = self.input_len
        H = self.output_len

        h_sum = torch.zeros(L, L)
        hw_sum = torch.zeros(L, H)
        for cd in self.client_data:
            h_sum.add_(cd["H_i"])
            hw_sum.add_(cd["H_i"] @ cd["W_i"])

        W_g = torch.linalg.solve(h_sum + self.gamma * torch.eye(L), hw_sum)
        self._load_linear_weights(self.model, W_g)

    def calculate_aggregation_weights(self) -> None:
        pass  # H_i encodes precision weighting; self.weights is unused


class FedDLSA_Client(FedRidge_Client):
    """Client for FedDLSA.

    Computes local OLS solution W_i and heteroscedasticity-aware precision
    matrix H_i = Sigma_xx_i / sigma_i^2 in a single data pass, where
    sigma_i^2 is estimated from the training residuals.

    No personalisation — receives and loads the global model only.

    Attributes
    ----------
    _w_i : torch.Tensor or None
        Local OLS solution, shape ``(L, H)``.
    _h_i : torch.Tensor or None
        Precision matrix H_i = Sigma_xx_i / sigma_i^2, shape ``(L, L)``.
    """

    _w_i: Optional[torch.Tensor] = None
    _h_i: Optional[torch.Tensor] = None

    def train(self) -> Dict[str, Any]:
        """Compute local OLS and precision matrix from the training data.

        Sets :attr:`_w_i` and :attr:`_h_i` in place (serial mode) and also
        returns them so :meth:`FedDLSA._apply_client_result` can propagate
        the results back in Ray-parallel mode.

        Returns
        -------
        dict
            ``{"w_i": Tensor[L, H], "h_i": Tensor[L, L], "train_samples": int}``
        """
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

        # Gaussian-NLL residual variance: RSS/n_obs via OLS optimality
        # trace(W_i^T Sxx W_i) = trace(W_i^T Sxy) at the optimum
        yy_mean = yy_sum / n_obs
        fit_trace = (w_i * sxy).sum().item()
        sigma_sq = max((yy_mean - fit_trace) / H, 1e-8)

        self._w_i = w_i
        self._h_i = sigma_xx / sigma_sq

        return {
            "w_i": self._w_i,
            "h_i": self._h_i,
            "train_samples": n_obs,
        }

    def variables_to_be_sent(self) -> Dict[str, Any]:
        return {
            "W_i": self._w_i,
            "H_i": self._h_i,
        }
