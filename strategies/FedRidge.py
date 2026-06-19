import time
from typing import Any, Dict, Optional

import torch

from .tFL import tFL, tFL_Client


class _LinearWeightsMixin:
    """Shared mixin that provides _load_linear_weights for server and client classes."""

    @staticmethod
    def _load_linear_weights(model: torch.nn.Module, W: torch.Tensor) -> None:
        """Load W (L×H) into the last nn.Linear(L,H) layer of model."""
        H, L = W.shape[1], W.shape[0]
        target = None
        for module in model.modules():
            if isinstance(module, torch.nn.Linear) and module.weight.shape == (H, L):
                target = module
        if target is not None:
            with torch.no_grad():
                target.weight.data.copy_(W.T.to(target.weight.device))
                if target.bias is not None:
                    target.bias.data.zero_()


class FedRidge(_LinearWeightsMixin, tFL):
    """
    FedRidge: One-Shot Federated Ridge Regression (arXiv:2601.08216) applied to LTSF.

    Clients upload full Sigma_xx (L×L) and Sigma_xy (L×H). Server aggregates
    and solves the global ridge regression exactly in one round.
    """

    optional = {"gamma": 0.1}

    @classmethod
    def args_update(cls, parser):
        parser.add_argument(
            "--gamma",
            type=float,
            default=None,
            help="Ridge regularization for OLS and personalization.",
        )
        return parser

    # ------------------------------------------------------------------ train

    def train(self) -> None:
        self.current_iter = 0
        self.logger.info(
            "%s: one-shot sufficient-statistics FL", self.__class__.__name__
        )
        round_start = time.time()

        self.selected_clients = [c for c in self.clients if not c.is_new]
        self.train_clients()
        self.receive_from_clients()
        self.calculate_aggregation_weights()
        self.aggregate_models()

        self.send_to_clients()

        for dataset_type in ["train", "test"]:
            if dataset_type == "train" and self.skip_eval_train:
                continue
            if not self.exclude_server_model_processes:
                self.evaluate_generalization_loss(dataset_type)
            self._pre_eval_hook(dataset_type)

        self.metrics["time_per_iter"].append(time.time() - round_start)
        self.save_models(save_type="best")
        self.fix_results()
        self.post_process()

    # --------------------------------------------------- federated FL hooks

    def _apply_client_result(
        self, client, package: Optional[Dict[str, Any]]
    ) -> None:
        """Apply sufficient statistics returned by :meth:`FedRidge_Client.train`.

        Called by :meth:`~tFL._dispatch` in Ray-parallel mode to write the
        statistics computed on the remote copy back to the original client.
        Serial mode does not need this because :meth:`FedRidge_Client.train`
        already sets the attributes in place.

        Parameters
        ----------
        client : FedRidge_Client
            The original client object whose statistics will be updated.
        package : dict or None
            Return value of ``FedRidge_Client.train()``, containing
            ``sigma_xx``, ``sigma_xy``, and ``train_samples``.
        """
        if package is None:
            return
        client._sigma_xx = package["sigma_xx"]
        client._sigma_xy = package["sigma_xy"]
        client.train_samples = package["train_samples"]

    def aggregate_models(self) -> None:
        L = self.input_len
        H = self.output_len

        sigma_xx_g = torch.zeros(L, L)
        sigma_xy_g = torch.zeros(L, H)
        for cd, w in zip(self.client_data, self.weights):
            sigma_xx_g.add_(cd["sigma_xx"], alpha=w.item())
            sigma_xy_g.add_(cd["sigma_xy"], alpha=w.item())

        W = torch.linalg.solve(sigma_xx_g + self.gamma * torch.eye(L), sigma_xy_g)

        self.sigma_xx_g = sigma_xx_g
        self.sigma_xy_g = sigma_xy_g
        self._load_linear_weights(self.model, W)


class FedRidge_Client(_LinearWeightsMixin, tFL_Client):
    """Client for FedRidge.

    Computes the local sufficient statistics ``Sigma_xx`` (L × L) and
    ``Sigma_xy`` (L × H) from the training data and uploads them to the
    server.  The server aggregates the weighted statistics and solves the
    global ridge regression in one round.

    Attributes
    ----------
    _sigma_xx : torch.Tensor or None
        Sample-normalised input covariance, shape ``(L, L)``.
    _sigma_xy : torch.Tensor or None
        Sample-normalised input-output cross-covariance, shape ``(L, H)``.
    """

    _sigma_xx: Optional[torch.Tensor] = None
    _sigma_xy: Optional[torch.Tensor] = None

    def train(self) -> Dict[str, Any]:
        """Compute local sufficient statistics from the training data.

        Sets :attr:`_sigma_xx` and :attr:`_sigma_xy` in place (consumed by
        :meth:`variables_to_be_sent` in serial mode) and returns the same
        values so :meth:`FedRidge._apply_client_result` can propagate them
        back to the original client object in Ray-parallel mode.

        Returns
        -------
        dict
            ``{"sigma_xx": Tensor[L, L], "sigma_xy": Tensor[L, H],
            "train_samples": int}``
        """
        loader = self.load_train_data()
        L = self.input_len
        H = self.output_len

        sigma_xx = torch.zeros(L, L)
        sigma_xy = torch.zeros(L, H)
        for batch_x, batch_y, *_ in loader:
            B, _, C = batch_x.shape
            x = batch_x.permute(0, 2, 1).reshape(B * C, L)
            y = batch_y.permute(0, 2, 1).reshape(B * C, H)
            sigma_xx.add_(x.T @ x)
            sigma_xy.add_(x.T @ y)

        N = self.train_samples
        self._sigma_xx = sigma_xx / N
        self._sigma_xy = sigma_xy / N

        return {
            "sigma_xx": self._sigma_xx,
            "sigma_xy": self._sigma_xy,
            "train_samples": N,
        }

    def variables_to_be_sent(self) -> Dict[str, Any]:
        return {
            "sigma_xx": self._sigma_xx,
            "sigma_xy": self._sigma_xy,
            "score": self.train_samples,
        }
