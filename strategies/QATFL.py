import math
import torch
from collections import OrderedDict
from torch.func import functional_call

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
    """Paper's vector quantizer Q(w) applied during the QAT forward pass.

    Exact Eqs. 7-10: scale = (v_max-v_min)/(q_max-q_min) (Eq. 9), zero point
    z = q_max - round(v_max/scale) (Eq. 10), stochastic rounding (Eq. 7),
    dequantize v' = (q-z)*scale (Eq. 8). q_min=-s//2, q_max=s//2-1 matches the
    paper's own 8-bit example (s=256 -> q_max=127, q_min=-128).
    """
    if s <= 0:
        return w
    v_max, v_min = w.max(), w.min()
    if v_max == v_min:
        return w
    q_min = -(s // 2)
    q_max = s // 2 - 1
    scale = (v_max - v_min) / (q_max - q_min)
    z = q_max - torch.round(v_max / scale)
    scaled = w / scale + z
    floor_val = torch.floor(scaled)
    prob = scaled - floor_val
    rand = torch.rand_like(w)
    q = torch.where(rand < prob, floor_val + 1.0, floor_val)
    q = torch.clamp(q, q_min, q_max)
    return (q - z) * scale


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

    def aggregate_client_updates(self, packages: "OrderedDict[int, dict]") -> None:
        """Unweighted FedAvg per paper Eq. 22 / Algorithm 1:

            w_{k+1} = w_k + (1/r) * sum_{i in S_k} Q_de(Q(w_{k,tau+M}^i - w_k))

        r = number of selected nodes -- NOT sample-count weighted.
        """
        r = len(packages)
        new_params = OrderedDict()
        for name in self.public_model_params:
            stacked = torch.stack(
                [p["regular_model_params"][name] for p in packages.values()], dim=-1
            )
            new_params[name] = stacked.sum(dim=-1) / r
        self._commit_global(new_params)


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
        # Paper's Table 1 tests tau=0 (all-QAT) as a valid config -- no floor here.
        qat_epochs = min(self.M_qat, self.epochs)
        regular_epochs = self.epochs - qat_epochs
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
        """One QAT epoch (paper Eqs. 20-21): loss computed via fake-quantized
        forward Q_fake(w) with straight-through gradient, but the optimizer
        update lands on the true persistent full-precision w -- quantization
        never overwrites the parameter itself, so noise cannot compound
        irreversibly across batches.
        """
        model.to(device)
        SharedMethods._move_optimizer_state_to_param_devices(optimizer)
        model.train()
        for batch_x, batch_y, x_mark, y_mark in dataloader:
            optimizer.zero_grad(set_to_none=True)
            batch_x = batch_x.to(device=device, dtype=torch.float32, non_blocking=True)
            batch_y = batch_y.to(device=device, dtype=torch.float32, non_blocking=True)
            x_mark = x_mark.to(device=device, dtype=torch.float32, non_blocking=True)
            y_mark = y_mark.to(device=device, dtype=torch.float32, non_blocking=True)

            # STE: forward value = Q_fake(w), but d(quantized)/dw = 1 (identity),
            # so the gradient accumulated on the true param w is unaffected.
            quantized = {
                name: p + (_fake_quantize_weight(p.detach(), self.s) - p).detach()
                for name, p in model.named_parameters()
                if p.requires_grad
            }
            outputs = functional_call(
                model, quantized, (batch_x,), {"x_mark": x_mark, "y_mark": y_mark}
            )
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
