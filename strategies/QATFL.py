import math
from typing import Any, Dict

import torch
from torch.func import functional_call

from .qFL import qFL, qFL_Client


class QATFLShared:
    """Affine stochastic quantization shared by QATFL server and client."""

    @staticmethod
    def quantize_tensor(tensor: torch.Tensor, levels: int) -> torch.Tensor:
        """Apply stochastic affine quantization."""
        if levels <= 0:
            return tensor
        if levels < 2 or levels & (levels - 1):
            raise ValueError("QATFL s must be 0 or a power-of-two level count")
        value_max, value_min = tensor.max(), tensor.min()
        if value_max == value_min:
            return tensor
        # Paper Eqs. 7-10: affine scale, zero point, stochastic round, dequantize.
        quant_min, quant_max = -(levels // 2), levels // 2 - 1
        scale = (value_max - value_min) / (quant_max - quant_min)
        zero = quant_max - torch.round(value_max / scale)
        scaled = tensor / scale + zero
        lower = scaled.floor()
        quantized = lower + (torch.rand_like(tensor) < scaled - lower)
        return (quantized.clamp(quant_min, quant_max) - zero) * scale

    @staticmethod
    def quantized_uplink_mb(
        model_params: Dict[str, torch.Tensor],
        levels: int,
    ) -> float:
        dimensions = sum(param.numel() for param in model_params.values())
        value_bits = math.ceil(math.log2(levels))
        return (dimensions * value_bits + 64) / 8 / (1024**2)


class QATFL(QATFLShared, qFL):
    """QAT-FL: Quantization-Aware Training for Federated Learning."""

    optional = {
        "s": 16,
        "M_qat": 2,
    }

    @classmethod
    def args_update(cls, parser: Any) -> Any:
        parser.add_argument(
            "-s",
            "--s_levels",
            dest="s",
            type=int,
            default=None,
            help="Number of quantization levels for QAT-FL (0 = disabled)",
        )
        parser.add_argument(
            "--M_qat",
            type=int,
            default=None,
            help="Number of QAT fine-tuning epochs per round",
        )
        return parser


class QATFL_Client(QATFLShared, qFL_Client):
    s: int = 16
    M_qat: int = 2

    def fit(self) -> None:
        self._set_worker_seed(seed=self._loader_seed(dataset_type="train"))
        loader = self.load_train_data()
        self.initialize_scheduler(steps_per_epoch=len(loader))
        offload_after_epoch = self.efficiency == "low"

        # Phase 1: regular local SGD updates (τ epochs)
        # Paper's Table 1 tests tau=0 (all-QAT) as a valid config -- no floor here.
        qat_epochs = min(self.M_qat, self.epochs)
        regular_epochs = self.epochs - qat_epochs
        for _ in range(regular_epochs):
            self.train_one_epoch(
                model=self.model,
                dataloader=loader,
                optimizer=self.optimizer,
                criterion=self.loss,
                scheduler=self.scheduler,
                device=self.device,
                offload_after=offload_after_epoch,
            )

        # Phase 2: QAT fine-tuning (M epochs) with fake quantization + STE
        for _ in range(qat_epochs):
            self._train_one_epoch_qat(
                model=self.model,
                dataloader=loader,
                optimizer=self.optimizer,
                criterion=self.loss,
                device=self.device,
                offload_after=offload_after_epoch,
            )

        if self.efficiency == "med":
            self.model.to("cpu")

    def _train_one_epoch_qat(
        self,
        model: Any,
        dataloader: Any,
        optimizer: Any,
        criterion: Any,
        device: Any,
        offload_after: Any = True,
    ) -> None:
        """Run one quantization-aware training epoch."""
        model.to(device)
        self._move_optimizer_state_to_param_devices(optimizer=optimizer)
        model.train()
        for batch_x, batch_y, x_mark, y_mark in dataloader:
            optimizer.zero_grad(set_to_none=True)
            batch_x = batch_x.to(device=device, dtype=torch.float32, non_blocking=True)
            batch_y = batch_y.to(device=device, dtype=torch.float32, non_blocking=True)
            x_mark = x_mark.to(device=device, dtype=torch.float32, non_blocking=True)
            y_mark = y_mark.to(device=device, dtype=torch.float32, non_blocking=True)

            # Paper Eqs. 20-21: forward Q_fake(w), with identity STE gradient;
            # so the gradient accumulated on the true param w is unaffected.
            quantized = {
                name: p
                + (self.quantize_tensor(tensor=p.detach(), levels=self.s) - p).detach()
                for name, p in model.named_parameters()
                if p.requires_grad
            }
            outputs = functional_call(
                model, quantized, (batch_x,), {"x_mark": x_mark, "y_mark": y_mark}
            )
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            self.step_scheduler_batch(
                scheduler=self.scheduler,
                batch_data=batch_x,
            )

        if self.s > 0 and offload_after:
            model.to("cpu")
        self.step_scheduler_epoch(scheduler=self.scheduler)
