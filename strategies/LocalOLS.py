import time

import ray
import torch

from .FedRidge import _LinearWeightsMixin
from .LocalOnly import LocalOnly, LocalOnly_Client


class LocalOLS(LocalOnly):
    """
    LocalOLS: Local-only one-shot ridge-OLS — no federation.

    Inherits LocalOnly's no-FL infrastructure and eval hooks
    (personalization loss only, no global model evaluation).
    gamma=0.0 gives plain OLS; gamma>0 gives ridge.

    Overrides train() to be one-shot: clients solve once, then evaluate.
    Re-running OLS on fixed data is idempotent, so multi-round iteration
    is wasteful.
    """

    optional = {"gamma": 0.0}

    def train(self) -> None:
        self.logger.info("%s: one-shot local OLS", self.__class__.__name__)
        round_start = time.time()
        self.current_iter = 0
        self.selected_clients = [i for i in range(self.num_clients) if not self.is_new[i]]
        packages = self.trainer.train(self.selected_clients)
        uplink, downlink = self._compute_send_mb(packages)
        self.metrics["downlink_mb"].append(downlink)
        for cid, mb in uplink.items():
            self._ensure_client_row(cid)["uplink_mb"][-1] = mb
        for cid, pkg in packages.items():
            self.clients_personal_model_params[cid].update(pkg["regular_model_params"])

        for dataset_type in ["train", "test"]:
            if dataset_type == "train" and self.skip_eval_train:
                continue
            self._pre_eval_hook(dataset_type)

        self.metrics["time_per_iter"].append(time.time() - round_start)
        self.fix_results(default=self.default_value)
        self._save_last_hook()
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


class LocalOLS_Client(_LinearWeightsMixin, LocalOnly_Client):
    """
    Client for LocalOLS.

    Replaces gradient training with a closed-form OLS solve each round.
    Data never changes, so recomputing is idempotent.
    """

    def fit(self) -> None:
        self._set_worker_seed(self._loader_seed("train"))
        loader = self.load_train_data()
        L = self.input_len
        H = self.output_len
        sxx = torch.zeros(L, L)
        sxy = torch.zeros(L, H)
        for batch_x, batch_y, *_ in loader:
            B, _, C = batch_x.shape
            x = batch_x.permute(0, 2, 1).reshape(B * C, L)
            y = batch_y.permute(0, 2, 1).reshape(B * C, H)
            sxx.add_(x.T @ x)
            sxy.add_(x.T @ y)

        N = self.train_samples
        W_i = torch.linalg.solve(sxx / N + self.gamma * torch.eye(L), sxy / N)
        self._load_linear_weights(self.model, W_i)
