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


# Paper sec:high_dim: for d > 1000 the dense O(d^2) Gram is prohibitive, so each
# client projects onto a shared R with i.i.d. N(0, 1/m) entries and sends the
# m x m Gram instead. R is regenerated from a fixed seed on every side, so it is
# identical everywhere and never goes on the wire.
_PROJ_MIN_DIM = 1000
_PROJ_SEED = 20260922


def _projection(L: int) -> Optional[torch.Tensor]:
    """Shared R (L x m), or None when the dense path applies."""
    if L <= _PROJ_MIN_DIM:
        return None
    m = int(0.4 * L)  # paper Exp. 7: the knee, ~5% MSE cost
    g = torch.Generator().manual_seed(_PROJ_SEED)
    return torch.randn(L, m, generator=g) / (m ** 0.5)


class FedRidge(_LinearWeightsMixin, tFL):
    """FedRidge: One-Shot Federated Ridge Regression (arXiv:2601.08216) applied to LTSF."""

    optional = {"gamma": 0.1}

    @classmethod
    def args_update(cls, parser: Any) -> Any:
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
        self.selected_clients = [
            i for i in range(self.num_clients) if not self.is_new[i]
        ]
        packages = self.trainer.train(self.selected_clients)
        uplink, downlink = self._compute_send_mb(packages=packages)
        self.metrics["downlink_mb"].append(downlink)
        for cid, mb in uplink.items():
            self._round_client_data.setdefault(cid, {})["uplink_mb"] = mb
        self.aggregate_client_updates(packages=packages)

        for dataset_type in ["train", "test"]:
            if dataset_type == "train" and self.skip_eval_train:
                continue
            if not self.exclude_server_model_processes:
                self.evaluate_generalization(dataset_type=dataset_type)
            self._pre_eval_hook(dataset_type=dataset_type)

        self.metrics["time_per_iter"].append(time.time() - round_start)
        self._save_best_hook()
        self._flush_round()
        self._save_last_hook()
        try:
            self.close_logger()
        except Exception:
            pass
        try:
            ray.shutdown()
        except Exception:
            pass

    def aggregate_client_updates(self, packages: Any) -> None:
        L = self.input_len
        H = self.output_len
        R = _projection(L)
        d = L if R is None else R.shape[1]

        # Paper Alg. 1: G = Σ G_k, h = Σ h_k (plain sums)
        sigma_xx_g = torch.zeros(d, d)
        sigma_xy_g = torch.zeros(d, H)
        for cid in packages:
            sigma_xx_g.add_(packages[cid]["sigma_xx"])
            sigma_xy_g.add_(packages[cid]["sigma_xy"])

        W = torch.linalg.solve(sigma_xx_g + self.gamma * torch.eye(d), sigma_xy_g)
        if R is not None:
            W = R @ W  # back to the L x H weight the linear head expects
        self.sigma_xx_g = sigma_xx_g
        self.sigma_xy_g = sigma_xy_g
        self._load_linear_weights(model=self.model, W=W)
        self._commit_global(
            new_params=OrderedDict(
                (k, v.detach().cpu().clone()) for k, v in self.model.named_parameters()
            )
        )


class FedRidge_Client(_LinearWeightsMixin, tFL_Client):
    """Client for FedRidge."""

    _sigma_xx: Optional[torch.Tensor] = None
    _sigma_xy: Optional[torch.Tensor] = None

    def fit(self) -> None:
        self._set_worker_seed(seed=self._loader_seed(dataset_type="train"))
        loader = self.load_train_data()
        L = self.input_len
        H = self.output_len

        R = _projection(L)
        d = L if R is None else R.shape[1]

        sigma_xx = torch.zeros(d, d)
        sigma_xy = torch.zeros(d, H)
        for batch_x, batch_y, *_ in loader:
            B, _, C = batch_x.shape
            x = batch_x.permute(0, 2, 1).reshape(B * C, L)
            y = batch_y.permute(0, 2, 1).reshape(B * C, H)
            if R is not None:
                x = x @ R
            sigma_xx.add_(x.T @ x)
            sigma_xy.add_(x.T @ y)

        self._sigma_xx = sigma_xx
        self._sigma_xy = sigma_xy

    def package(self) -> Dict[str, Any]:
        result = super().package()
        result["sigma_xx"] = self._sigma_xx
        result["sigma_xy"] = self._sigma_xy
        result["__wire__"] = ("sigma_xx", "sigma_xy")
        return result
