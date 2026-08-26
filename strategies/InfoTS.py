from typing import Any

import numpy as np
import torch

from .nFL import nFL, nFL_Client


class InfoTS(nFL):
    """InfoTS meta-contrastive pretraining followed by frozen ridge fitting."""

    compulsory = {"model": "InfoTS"}
    optional = {
        "pretrain_epochs": 400,
        "pretrain_lr": 1e-3,
        "pretrain_meta_lr": 1e-2,
        "pretrain_meta_epoch": 2,
        "pretrain_temp_t0": 2.0,
        "pretrain_temp_t1": 0.1,
        "ridge_alpha": 1.0,
    }

    @classmethod
    def args_update(cls, parser: Any) -> None:
        parser.add_argument("--pretrain_epochs", type=int, default=None)
        parser.add_argument("--pretrain_lr", type=float, default=None)
        parser.add_argument("--pretrain_meta_lr", type=float, default=None)
        parser.add_argument("--pretrain_meta_epoch", type=int, default=None)
        parser.add_argument("--pretrain_temp_t0", type=float, default=None)
        parser.add_argument("--pretrain_temp_t1", type=float, default=None)
        parser.add_argument("--ridge_alpha", type=float, default=None)


class InfoTS_Client(nFL_Client):
    def fit(self) -> None:
        self._set_worker_seed(seed=self._loader_seed(dataset_type="train"))

        train_loader = self.load_train_data()

        self.model.to(self.device)
        if not self.auxiliary_state.get("infots_pretrained"):
            if self.pretrain_epochs > 0:
                self._pretrain(train_loader=train_loader)
            self.fit_ridge_head(
                model=self.model,
                dataloader=train_loader,
                device=self.device,
                alpha=self.ridge_alpha,
            )
            self.auxiliary_state["infots_pretrained"] = True

        if self.efficiency != "high":
            self.model.to("cpu")

    def _pretrain(self, train_loader: Any) -> None:
        """Self-supervised pre-training: alternates updates of encoder and AutoAUG."""
        encoder_opt = torch.optim.Adam(
            self.model.encoder.parameters(), lr=self.pretrain_lr, betas=(0.9, 0.999)
        )
        meta_opt = torch.optim.Adam(
            self.model.aug.parameters(), lr=self.pretrain_meta_lr, betas=(0.9, 0.999)
        )
        meta_head_opt = torch.optim.Adam(
            self.model.meta_unsup_head.parameters(),
            lr=self.pretrain_meta_lr,
            betas=(0.9, 0.999),
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

            if (epoch + 1) % self.pretrain_meta_epoch == 0:
                for batch_x, *_ in train_loader:
                    batch_x = batch_x.to(
                        self.device, dtype=torch.float32, non_blocking=True
                    )
                    if batch_x.size(0) == self.batch_size:
                        self.model.meta_step(
                            batch_x, meta_opt, meta_head_opt, temperature=temperature
                        )

            for batch_x, *_ in train_loader:
                batch_x = batch_x.to(
                    self.device, dtype=torch.float32, non_blocking=True
                )
                encoder_opt.zero_grad(set_to_none=True)
                loss = self.model.pretrain_loss(batch_x, temperature=temperature)
                loss.backward()
                encoder_opt.step()
