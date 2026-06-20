import time
from collections import OrderedDict
from typing import Any, Dict, Optional

import ray
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

    def train(self) -> None:
        self.logger.info(
            "%s: one-shot sufficient-statistics FL", self.__class__.__name__
        )
        round_start = time.time()
        self.current_iter = 0
        self.selected_clients = [i for i in range(self.num_clients) if not self.is_new[i]]
        self.metrics["send_mb"].append(self._send_mb_per_round)
        packages = self.trainer.train(self.selected_clients)
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
        cids = list(packages.keys())
        scores = [packages[cid]["score"] for cid in cids]
        total = float(sum(scores))

        sigma_xx_g = torch.zeros(L, L)
        sigma_xy_g = torch.zeros(L, H)
        for cid, w in zip(cids, [s / total for s in scores]):
            sigma_xx_g.add_(packages[cid]["sigma_xx"], alpha=w)
            sigma_xy_g.add_(packages[cid]["sigma_xy"], alpha=w)

        W = torch.linalg.solve(sigma_xx_g + self.gamma * torch.eye(L), sigma_xy_g)
        self.sigma_xx_g = sigma_xx_g
        self.sigma_xy_g = sigma_xy_g
        self._load_linear_weights(self.model, W)
        self._commit_global(
            OrderedDict(
                (k, v.detach().cpu().clone()) for k, v in self.model.named_parameters()
            )
        )


class FedRidge_Client(_LinearWeightsMixin, tFL_Client):
    """Client for FedRidge.

    Computes the local sufficient statistics Sigma_xx (L × L) and
    Sigma_xy (L × H) from the training data and uploads them to the
    server.  The server aggregates the weighted statistics and solves the
    global ridge regression in one round.
    """

    _sigma_xx: Optional[torch.Tensor] = None
    _sigma_xy: Optional[torch.Tensor] = None

    def fit(self) -> None:
        self._set_worker_seed(self._loader_seed("train"))
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

    def package(self, train_time: float) -> Dict[str, Any]:
        result = super().package(train_time)
        result["sigma_xx"] = self._sigma_xx
        result["sigma_xy"] = self._sigma_xy
        return result
