import time
from typing import Any, Dict, Optional

import torch

from .pFL import pFL, pFL_Client


class FedRidge(pFL):
    """
    FedRidge: One-Shot Federated Ridge Regression (arXiv:2601.08216) applied to LTSF.

    Clients upload full Sigma_xx (L×L) and Sigma_xy (L×H). Server aggregates
    and solves the global ridge regression exactly in one round. Clients
    personalize by blending local and global covariance matrices.

    This one-shot strategy uses sufficient statistics instead of iterative
    model averaging.
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

        self.selected_clients = list(self.clients)
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

    def train_clients(self) -> None:
        for client in self.clients:
            client.compute_statistics()

    def aggregate_models(self) -> None:
        L = self.input_len
        H = self.output_len
        gamma = getattr(self, "gamma", 0.1)

        Sigma_xx_g = torch.zeros(L, L)
        Sigma_xy_g = torch.zeros(L, H)
        for cd, w in zip(self.client_data, self.weights):
            Sigma_xx_g.add_(cd["Sigma_xx"], alpha=w.item())
            Sigma_xy_g.add_(cd["Sigma_xy"], alpha=w.item())

        W = torch.linalg.solve(Sigma_xx_g + gamma * torch.eye(L), Sigma_xy_g)

        self.Sigma_xx_g = Sigma_xx_g
        self.Sigma_xy_g = Sigma_xy_g
        _load_linear_weights(self.model, W)

    def variables_to_be_sent(self) -> Dict[str, Any]:
        base = super().variables_to_be_sent()
        base["Sigma_xx_g"] = getattr(self, "Sigma_xx_g", None)
        base["Sigma_xy_g"] = getattr(self, "Sigma_xy_g", None)
        return base


class FedRidge_Client(pFL_Client):
    """
    Client for FedRidge.

    Computes full Sigma_xx and Sigma_xy from local data. Personalizes the
    received global model via ridge blend of local and global statistics.
    """

    _Sigma_xx: Optional[torch.Tensor] = None
    _Sigma_xy: Optional[torch.Tensor] = None
    Sigma_xx_g: Optional[torch.Tensor] = None
    Sigma_xy_g: Optional[torch.Tensor] = None

    def compute_statistics(self) -> None:
        loader = self.load_train_data()
        L = self.input_len
        H = self.output_len

        Sigma_xx = torch.zeros(L, L)
        Sigma_xy = torch.zeros(L, H)
        for batch_x, batch_y, *_ in loader:
            B, _, C = batch_x.shape
            x = batch_x.permute(0, 2, 1).reshape(B * C, L)
            y = batch_y.permute(0, 2, 1).reshape(B * C, H)
            Sigma_xx.add_(x.T @ x)
            Sigma_xy.add_(x.T @ y)

        N = self.train_samples
        self._Sigma_xx = Sigma_xx / N
        self._Sigma_xy = Sigma_xy / N

    def variables_to_be_sent(self) -> Dict[str, Any]:
        return {
            "Sigma_xx": self._Sigma_xx,
            "Sigma_xy": self._Sigma_xy,
            "score": self.train_samples,
        }

    def receive_from_server(self, data: Dict[str, Any]) -> None:
        self.Sigma_xx_g = data.pop("Sigma_xx_g", None)
        self.Sigma_xy_g = data.pop("Sigma_xy_g", None)
        super().receive_from_server(data)
        if (
            self.Sigma_xx_g is not None
            and self._Sigma_xx is not None
            and self._Sigma_xy is not None
        ):
            self._personalize()

    def _personalize(self) -> None:
        gamma = getattr(self, "gamma", 0.1)
        L = self.input_len
        A = self._Sigma_xx + gamma * self.Sigma_xx_g + gamma * torch.eye(L)
        b = self._Sigma_xy + gamma * self.Sigma_xy_g
        W_pers = torch.linalg.solve(A, b)
        _load_linear_weights(self.model, W_pers)

    def train(self) -> Optional[Dict[str, Any]]:
        return None


# ------------------------------------------------------------------ utilities


def _load_linear_weights(model: torch.nn.Module, W: torch.Tensor) -> None:
    """Load OLS solution W (L, H) into the model's first linear layer."""
    H, L = W.shape[1], W.shape[0]
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim == 2 and param.shape == (H, L):
                param.data.copy_(W.T.to(param.device))
            elif param.ndim == 1 and param.shape[0] == H:
                param.data.zero_()
