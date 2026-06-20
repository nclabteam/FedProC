import numpy as np
import torch

from .pFL import pFL, pFL_Client


class InfoTS(pFL):
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

    compulsory = {"model": "InfoTS"}
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

    def aggregate_client_updates(self, packages) -> None:
        for cid, pkg in packages.items():
            self.clients_personal_model_params[cid].update(pkg["regular_model_params"])

    def evaluate_generalization(self, *args, **kwargs):
        pass


class InfoTS_Client(pFL_Client):
    def fit(self) -> None:
        self._set_worker_seed(self._loader_seed("train"))

        train_loader = self.load_train_data()

        self.model.to(self.device)

        if hasattr(self.model, "pretrain_loss") and self.pretrain_epochs > 0:
            pretrain_loader = self.load_train_data()
            self._pretrain(pretrain_loader)

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
