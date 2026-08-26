from typing import Any

import torch

from .nFL import nFL, nFL_Client


class SimTS(nFL):
    """Local SimTS representation pretraining followed by frozen ridge fitting."""

    compulsory = {"model": "SimTS"}
    optional = {
        "pretrain_epochs": 500,
        "pretrain_lr": 1e-3,
        "ridge_alpha": 1.0,
    }

    @classmethod
    def args_update(cls, parser: Any) -> None:
        parser.add_argument("--pretrain_epochs", type=int, default=None)
        parser.add_argument("--pretrain_lr", type=float, default=None)
        parser.add_argument("--ridge_alpha", type=float, default=None)


class SimTS_Client(nFL_Client):

    def fit(self) -> None:
        self._set_worker_seed(seed=self._loader_seed(dataset_type="train"))
        train_loader = self.load_train_data()

        self.model.to(self.device)
        if not self.auxiliary_state.get("simts_pretrained"):
            if self.pretrain_epochs > 0:
                self._pretrain(train_loader=train_loader)
            self.fit_ridge_head(
                model=self.model,
                dataloader=train_loader,
                device=self.device,
                alpha=self.ridge_alpha,
            )
            self.auxiliary_state["simts_pretrained"] = True

        if self.efficiency != "high":
            self.model.to("cpu")

    def _pretrain(self, train_loader: Any) -> None:
        """Pretrain the encoder and predictor."""
        pretrain_opt = torch.optim.SGD(
            [
                {"params": list(self.model.encoder.parameters())},
                {
                    "params": list(self.model.predictor.parameters()),
                    "lr": self.pretrain_lr * 0.0001,
                },
            ],
            lr=self.pretrain_lr,
            momentum=0.9,
            weight_decay=1e-4,
        )
        self.model.train()
        for _ in range(self.pretrain_epochs):
            for batch_x, *_ in train_loader:
                batch_x = batch_x.to(
                    device=self.device, dtype=torch.float32, non_blocking=True
                )
                # Paper Algorithm 1: optimize the cosine prediction loss.
                loss = self.model.pretrain_loss(batch_x)
                pretrain_opt.zero_grad(set_to_none=True)
                loss.backward()
                pretrain_opt.step()
