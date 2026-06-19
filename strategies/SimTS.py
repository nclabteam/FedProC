import time
from typing import Any, Dict

import torch

from .nFL import nFL, nFL_Client


class SimTS(nFL):
    """
    SimTS: local self-supervised contrastive pre-training + supervised fine-tuning.

    Each client trains independently (no federation):
      Round t:
        1. Self-supervised pre-training — encoder + predictor optimised with the
           SimTS cosine similarity loss (pretrain_epochs epochs, pretrain_lr).
        2. Supervised fine-tuning — full model optimised with the task loss
           (epochs epochs, learning_rate).

    Designed to be paired with --model SimTS.  The model must expose a
    ``pretrain_loss(x)`` method; if it does not, the pre-training phase is
    silently skipped so the strategy degrades gracefully to pure local training.

    Reference: Zheng & Ma, "SimTS: Rethinking Contrastive Representation
    Learning for Time Series Forecasting", arXiv:2303.18205.
    Short version accepted at ICASSP 2024 (doi:10.1109/ICASSP48485.2024.10446875).
    """

    compulsory = {**nFL.compulsory, "model": "SimTS"}
    optional = {
        "pretrain_epochs": 10,
        "pretrain_lr": 1e-3,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--pretrain_epochs", type=int, default=None)
        parser.add_argument("--pretrain_lr", type=float, default=None)

    def evaluate_generalization_loss(self, *args, **kwargs):
        pass

    def _pre_eval_hook(self, dataset_type: str) -> None:
        self.evaluate_personalization_loss(dataset_type)


class SimTS_Client(nFL_Client):

    def train(self) -> Dict[str, Any]:
        seed = self._loader_seed("train") if hasattr(self, "_loader_seed") else None
        self._set_worker_seed(seed)
        train_loader = self.load_train_data()
        start_time = time.time()

        self.model.to(self.device)

        # Phase 1: self-supervised pre-training
        if hasattr(self.model, "pretrain_loss") and self.pretrain_epochs > 0:
            self._pretrain(train_loader)

        # Phase 2: supervised fine-tuning
        offload_after_epoch = self.efficiency == "low"
        for _ in range(self.epochs):
            self.train_one_epoch(
                model=self.model,
                dataloader=train_loader,
                optimizer=self.optimizer,
                criterion=self.loss,
                scheduler=self.scheduler,
                device=self.device,
                offload_after=offload_after_epoch,
            )

        if self.efficiency == "med":
            self.model.to("cpu")

        return {
            "model": self.model,
            "optimizer_state": self.optimizer,
            "train_time": time.time() - start_time,
            "train_samples": self.train_samples,
        }

    def _pretrain(self, train_loader):
        """Self-supervised phase: train encoder + predictor with cosine loss."""
        pretrain_opt = torch.optim.SGD(
            [
                {"params": list(self.model.encoder.parameters())},
                {
                    "params": list(self.model.predictor.parameters()),
                    "lr": self.pretrain_lr * 0.01,
                },
            ],
            lr=self.pretrain_lr,
        )
        self.model.train()
        for _ in range(self.pretrain_epochs):
            for batch_x, *_ in train_loader:
                batch_x = batch_x.to(
                    device=self.device, dtype=torch.float32, non_blocking=True
                )
                loss = self.model.pretrain_loss(batch_x)
                pretrain_opt.zero_grad(set_to_none=True)
                loss.backward()
                pretrain_opt.step()
