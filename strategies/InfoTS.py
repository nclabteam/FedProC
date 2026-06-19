import time
from typing import Any, Dict

import numpy as np
import torch

from .nFL import nFL, nFL_Client


class InfoTS(nFL):
    """
    InfoTS: Information-Aware Time Series Meta-Contrastive Learning.

    Each client trains locally and independently:
      Round t:
        1. Alternating self-supervised meta-pretraining:
           - Minimizes global and local InfoNCE contrastive losses over TSEncoder.
           - Optimizes Gumbel-Softmax augmentation selection weights (AutoAUG)
             and classifier heads via meta-steps.
        2. Supervised fine-tuning:
           - Tunes the forecasting model end-to-end on target MSE forecasting loss.
    """

    compulsory = {**nFL.compulsory, "model": "InfoTS"}
    optional = {
        "pretrain_epochs": 10,
        "pretrain_lr": 1e-3,
        "pretrain_meta_lr": 1e-2,
        "pretrain_meta_epoch": 2,
        "pretrain_temp_t0": 2.0,
        "pretrain_temp_t1": 0.1,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--pretrain_epochs", type=int, default=None)
        parser.add_argument("--pretrain_lr", type=float, default=None)
        parser.add_argument("--pretrain_meta_lr", type=float, default=None)
        parser.add_argument("--pretrain_meta_epoch", type=int, default=None)
        parser.add_argument("--pretrain_temp_t0", type=float, default=None)
        parser.add_argument("--pretrain_temp_t1", type=float, default=None)

    def evaluate_generalization_loss(self, *args, **kwargs):
        pass

    def _pre_eval_hook(self, dataset_type: str) -> None:
        self.evaluate_personalization_loss(dataset_type)


class InfoTS_Client(nFL_Client):
    def train(self) -> Dict[str, Any]:
        self._set_worker_seed(self._loader_seed("train"))

        train_loader = self.load_train_data()
        start_time = time.time()

        self.model.to(self.device)

        # Phase 1: self-supervised meta-pretraining
        if hasattr(self.model, "pretrain_loss") and self.pretrain_epochs > 0:
            pretrain_loader = self.load_train_data()
            self._pretrain(pretrain_loader)

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
        """Self-supervised pre-training: alternates updates of encoder and AutoAUG."""
        encoder_opt = torch.optim.AdamW(
            self.model.encoder.parameters(), lr=self.pretrain_lr
        )
        meta_opt = torch.optim.AdamW(
            self.model.aug.parameters(), lr=self.pretrain_meta_lr
        )
        meta_head_opt = torch.optim.AdamW(
            self.model.meta_unsup_head.parameters(), lr=self.pretrain_meta_lr
        )

        self.model.train()
        for epoch in range(self.pretrain_epochs):
            temperature = float(
                self.pretrain_temp_t0
                * np.power(
                    self.pretrain_temp_t1 / self.pretrain_temp_t0,
                    (epoch + 1) / self.pretrain_epochs,
                )
            )

            # Alternate meta update step
            if (epoch + 1) % self.pretrain_meta_epoch == 0:
                for batch_x, *_ in train_loader:
                    batch_x = batch_x.to(
                        self.device, dtype=torch.float32, non_blocking=True
                    )
                    if batch_x.size(0) == self.batch_size:
                        self.model.meta_step(
                            batch_x, meta_opt, meta_head_opt, temperature=temperature
                        )

            # Normal encoder contrastive update step
            for batch_x, *_ in train_loader:
                batch_x = batch_x.to(
                    self.device, dtype=torch.float32, non_blocking=True
                )
                encoder_opt.zero_grad(set_to_none=True)
                loss = self.model.pretrain_loss(batch_x, temperature=temperature)
                loss.backward()
                encoder_opt.step()
