import math
import torch
from collections import OrderedDict

from .tFL import tFL, tFL_Client
from .base import SharedMethods

_BYTES_PER_MB = 1024 ** 2


def _qsgd_uplink_mb(model_params: OrderedDict, s: int) -> float:
    """QSGD wire size per paper Eq. 11: (d*ceil(log2(s)) + d + 32) bits.

    d magnitudes at ceil(log2(s)) bits each, d sign bits, 32-bit norm scalar.
    """
    d = sum(p.numel() for p in model_params.values())
    n_bits = math.ceil(math.log2(max(s, 2)))
    total_bits = d * n_bits + d + 32
    return total_bits / 8 / _BYTES_PER_MB


def _fake_quantize_weight(w: torch.Tensor, s: int) -> torch.Tensor:
    """Symmetric per-layer fake quantization with max-based scale."""
    if s <= 0:
        return w
    scale = w.abs().max()
    if scale == 0:
        return w
    half = s // 2
    q = torch.clamp(torch.round(w / scale * half), -half, half - 1)
    return q * scale / half


class QATFLShared(SharedMethods):
    """Shared QAT-FL utilities: quantization and STE."""

    @staticmethod
    def quantize_tensor(tensor: torch.Tensor, s: int) -> torch.Tensor:
        """Stochastic quantization (QSGD-style, same as FedPAQ)."""
        norm = torch.norm(tensor)
        if norm == 0:
            return tensor
        abs_scaled = (torch.abs(tensor) / norm) * s
        l = torch.floor(abs_scaled)
        prob = abs_scaled - l
        rand = torch.rand_like(tensor)
        rounded = torch.where(rand < prob, l + 1.0, l)
        return norm * torch.sign(tensor) * (rounded / s)


class QATFL(QATFLShared, tFL):
    """QAT-FL: Quantization-Aware Training for Federated Learning.

    Reduces quantization distortion by adding fake-quantization modules
    during training so the model adapts to future quantization (paper 10025).

    Each round: τ regular SGD epochs + M_qat QAT fine-tuning epochs.
    Communication uses stochastic quantization of the update delta.
    """

    optional = {
        "s": 16,
        "M_qat": 2,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument(
            "-s", "--s_levels", dest="s", type=int, default=None,
            help="Number of quantization levels for QAT-FL (0 = disabled)",
        )
        parser.add_argument(
            "--M_qat", type=int, default=None,
            help="Number of QAT fine-tuning epochs per round",
        )
        return parser

    def _compute_send_mb(self, packages) -> tuple:
        # Uplink: QSGD quantized delta per paper Eq. 11 (not full float32 model)
        uplink_mb = _qsgd_uplink_mb(self.public_model_params, self.s)
        uplink = {cid: uplink_mb for cid in packages}
        # Downlink: full float32 global model w_k (Algorithm 1, line 2)
        downlink = sum(self._downlink_sizes.get(cid, 0.0) for cid in self.selected_clients)
        return uplink, downlink


class QATFL_Client(QATFLShared, tFL_Client):
    s: int = 16
    M_qat: int = 2

    def set_parameters(self, package: dict) -> None:
        super().set_parameters(package)
        self._init_params = {
            name: param.data.clone().cpu()
            for name, param in self.model.named_parameters()
        }

    def fit(self) -> None:
        SharedMethods._set_worker_seed(self._loader_seed("train"))
        loader = self.load_train_data()
        offload_after_epoch = self.efficiency == "low"

        # Phase 1: regular local SGD updates (τ epochs)
        qat_epochs = min(self.M_qat, max(0, self.epochs - 1))
        regular_epochs = max(0, self.epochs - qat_epochs)
        for _ in range(regular_epochs):
            self.train_one_epoch(
                model=self.model, dataloader=loader, optimizer=self.optimizer,
                criterion=self.loss, scheduler=self.scheduler, device=self.device,
                offload_after=offload_after_epoch,
            )

        # Phase 2: QAT fine-tuning (M epochs) with fake quantization + STE
        for _ in range(qat_epochs):
            self._train_one_epoch_qat(
                model=self.model, dataloader=loader, optimizer=self.optimizer,
                criterion=self.loss, device=self.device,
                offload_after=offload_after_epoch,
            )

        if self.efficiency == "med":
            self.model.to("cpu")

    def _train_one_epoch_qat(
        self, model, dataloader, optimizer, criterion, device,
        offload_after=True,
    ) -> None:
        """One QAT epoch: fake-quantize weights, forward, backward (STE), update."""
        model.to(device)
        SharedMethods._move_optimizer_state_to_param_devices(optimizer)
        model.train()
        for batch_x, batch_y, x_mark, y_mark in dataloader:
            optimizer.zero_grad(set_to_none=True)
            batch_x = batch_x.to(device=device, dtype=torch.float32, non_blocking=True)
            batch_y = batch_y.to(device=device, dtype=torch.float32, non_blocking=True)
            x_mark = x_mark.to(device=device, dtype=torch.float32, non_blocking=True)
            y_mark = y_mark.to(device=device, dtype=torch.float32, non_blocking=True)

            # Fake-quantize all trainable weights in-place (STE: backward uses identity)
            for param in model.parameters():
                if param.requires_grad:
                    param.data = _fake_quantize_weight(param.data, self.s)

            outputs = model(batch_x, x_mark=x_mark, y_mark=y_mark)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        if self.s > 0 and offload_after:
            model.to("cpu")

    def package(self) -> dict:
        result = super().package()
        if self.s > 0:
            quantized = {}
            for name, x_tau in result["regular_model_params"].items():
                delta = x_tau.float() - self._init_params[name].float()
                q_delta = self.quantize_tensor(delta, self.s)
                quantized[name] = (self._init_params[name].float() + q_delta).to(x_tau.dtype)
            result["regular_model_params"] = quantized
        return result
