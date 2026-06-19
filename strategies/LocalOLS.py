import time

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
        self.current_iter = 0
        self.logger.info("%s: one-shot local OLS", self.__class__.__name__)
        round_start = time.time()

        self.selected_clients = [c for c in self.clients if not c.is_new]
        self.train_clients()

        for dataset_type in ["train", "test"]:
            if dataset_type == "train" and self.skip_eval_train:
                continue
            self._pre_eval_hook(dataset_type)

        self.metrics["time_per_iter"].append(time.time() - round_start)
        self.fix_results()
        self.post_process()


class LocalOLS_Client(_LinearWeightsMixin, LocalOnly_Client):
    """
    Client for LocalOLS.

    Replaces gradient training with a closed-form OLS solve each round.
    Data never changes, so recomputing is idempotent.
    """

    def train(self):
        start_time = time.time()
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

        train_time = time.time() - start_time
        model = self._clone_model_to_cpu(self.model) if self.efficiency == "high" else self.model
        return {
            "model": model,
            "optimizer_state": self.optimizer,
            "train_time": train_time,
            "train_samples": self.train_samples,
        }

    def adapt(self, global_model=None) -> None:
        self.train()
