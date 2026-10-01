import json
import logging
import warnings
from argparse import Namespace
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
from scipy import stats
from torch.func import functional_call

from .tFL import tFL, tFL_Client


class FedHQShared:
    """INT8 symmetric per-tensor quantization shared by FedHQ server and client."""

    @staticmethod
    def quantize_int8(
        params: "dict[str, torch.Tensor]",
    ) -> "tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]":
        q, scales = OrderedDict(), OrderedDict()
        for name, tensor in params.items():
            tensor = tensor.detach().float().cpu()
            scale = (tensor.abs().max() / 127).clamp_min(1e-12)
            q[name] = torch.round(tensor / scale).clamp(-127, 127).to(torch.int8)
            scales[name] = scale
        return q, scales

    @staticmethod
    def dequantize_int8(
        q: "dict[str, torch.Tensor]", scales: "dict[str, torch.Tensor]"
    ) -> "OrderedDict[str, torch.Tensor]":
        return OrderedDict((name, q[name].float() * scales[name]) for name in q)

    @staticmethod
    def fake_quant(tensor: torch.Tensor) -> torch.Tensor:
        """INT8 quantize-dequantize with a straight-through gradient."""
        scale = (tensor.detach().abs().max() / 127).clamp_min(1e-12)
        quantized = torch.round(tensor.detach() / scale).clamp(-127, 127) * scale
        return tensor + (quantized - tensor).detach()


# Paper Tab. 2 candidates with exactly its parameters: (name, family, discrete,
# fixed scipy loc/scale). Unlisted loc/scale are pinned; the power-law
# normalizer c fixes its scale to the smallest observation.
_CANDIDATES = (
    ("normal", stats.norm, False, lambda v: {}),
    ("power-law", stats.pareto, False, lambda v: {"floc": 0, "fscale": v.min()}),
    ("binomial", stats.binom, True, None),
    ("poisson", stats.poisson, True, None),
    ("log-normal", stats.lognorm, False, lambda v: {"floc": 0}),
    ("student-t", stats.t, False, lambda v: {"floc": 0, "fscale": 1}),
    ("logistic", stats.logistic, False, lambda v: {}),
    ("beta", stats.beta, False, lambda v: {"floc": 0, "fscale": 1}),
    ("gamma", stats.gamma, False, lambda v: {"floc": 0}),
)


def ks_distance(values: np.ndarray, fitted: Any, discrete: bool) -> float:
    """sup |ECDF - F| checked on both sides of every jump, so ties and pmfs count."""
    points, counts = np.unique(values, return_counts=True)
    right = np.cumsum(counts) / len(values)
    left = right - counts / len(values)
    cdf = fitted.cdf(points)
    cdf_left = fitted.cdf(points - 1) if discrete else cdf
    return float(max(np.abs(right - cdf).max(), np.abs(left - cdf_left).max()))


def fit_distribution(values: np.ndarray) -> "tuple[str, float]":
    """Alg. 1: keep the candidate with the smallest KS statistic; return its mean.

    A candidate whose support excludes the data fails to fit and is skipped.
    """
    values = np.asarray(values, dtype=np.float64)
    integral = bool(np.all(values >= 0) and np.all(values == np.round(values)))
    best, best_name, best_mean = np.inf, None, None
    for name, family, discrete, fixed in _CANDIDATES:
        # A pmf on non-negative integers gives non-integer data zero likelihood.
        if discrete and not integral:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if discrete:
                    top = float(values.max()) + 1
                    bounds = (
                        {"n": (0, 2 * top), "p": (0, 1)}
                        if family is stats.binom
                        else {"mu": (0, top)}
                    )
                    bounds["loc"] = (0, 0)
                    fitted = family(
                        **stats.fit(family, values, bounds).params._asdict()
                    )
                else:
                    fitted = family(*family.fit(values, **fixed(values)))
                mean = float(fitted.mean())
                goodness = ks_distance(values, fitted, discrete)
        except Exception:
            continue
        if np.isfinite(mean) and np.isfinite(goodness) and goodness < best:
            best, best_name, best_mean = goodness, name, mean
    if best_name is None:
        raise ValueError("FedHQ: no candidate distribution fits this client's data")
    return best_name, best_mean


def normalize(values: np.ndarray) -> np.ndarray:
    """Eq. 5 min-max Norm; a constant vector carries no ranking, so 0.5."""
    low, high = values.min(), values.max()
    if high - low <= 0:
        return np.full_like(values, 0.5)
    return (values - low) / (high - low)


def assign_strategies(
    norm_speed: np.ndarray, norm_acc: np.ndarray, xi: float, area_threshold: float
) -> "list[str]":
    """Eq. 6 geometric segmentation.

    Triangle legs are NormSig^Speed (x) and NormSig^Acc (y); hypotenuse slope
    is acc/speed and Theta's slope is (1 - xi) / xi. PTQ when the hypotenuse is
    steeper, or when the triangle area is below the threshold.
    """
    out = []
    for speed, acc in zip(norm_speed, norm_acc):
        if 0.5 * speed * acc < area_threshold or xi * acc > (1 - xi) * speed:
            out.append("PTQ")
        else:
            out.append("QAT")
    return out


class FedHQ(FedHQShared, tFL):
    """FedHQ: hybrid runtime quantization (coarse-grained global initialization)."""

    # auto = FedHQ; ptq / qat / random = the paper's FedFQ / FedAQT / FedRH baselines.
    optional = {
        "xi": 0.2,
        "area_threshold": 0.0625,
        "sig_alpha": 0.5,
        "hq_mode": "auto",
    }

    @classmethod
    def args_update(cls, parser: Any) -> None:
        parser.add_argument("--xi", type=float, default=None)
        parser.add_argument("--area_threshold", type=float, default=None)
        parser.add_argument(
            "--sig_alpha", type=float, default=None, help="Eq. 3 alpha in (0, 1)"
        )
        parser.add_argument(
            "--hq_mode",
            type=str,
            default=None,
            choices=["auto", "ptq", "qat", "random"],
        )
        parser.add_argument(
            "--hardware",
            type=float,
            nargs="+",
            default=None,
            help="per-client H_m(alpha M, beta C) of Eq. 1; default uniform",
        )

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        if not 0 < self.sig_alpha < 1:
            raise ValueError("FedHQ sig_alpha must be in (0, 1)")
        hardware = getattr(self, "hardware", None) or [1.0] * self.num_clients
        if len(hardware) != self.num_clients:
            raise ValueError("FedHQ --hardware needs one value per client")
        with open(self.path_info, "r", encoding="utf-8") as f:
            info = json.load(f)[: self.num_clients]
        n_params = sum(p.numel() for p in self.model.parameters())
        sig_speed, sig_acc, fitted = [], [], []
        for client, h in zip(info, hardware):
            with np.load(client["paths"]["train"]) as data:
                x = data["x"].astype(np.float64)
            volume = len(x)
            # Eq. 1 with B_m = batch_size and E_m = epochs for every client.
            sig_speed.append(volume / self.batch_size * self.epochs * n_params / h)
            # Alg. 1 on the series values (first step of each stride-1 window);
            # Eq. 3 sums over those same V points: V^-alpha * sum_v |data_v - mean|.
            points = x[:, 0, :].reshape(-1)
            name, mean = fit_distribution(points)
            fitted.append(name)
            sig_acc.append(
                len(points) ** (-self.sig_alpha) * np.abs(points - mean).sum()
            )
        sig_speed = np.asarray(sig_speed)
        span = sig_speed.max() - sig_speed.min()
        # Eq. 2 is undefined when every T_m is equal; Norm handles that case.
        sig_speed = sig_speed / span if span > 0 else sig_speed
        norm_speed, norm_acc = normalize(sig_speed), normalize(np.asarray(sig_acc))
        if self.hq_mode == "auto":
            self.client_strategy = assign_strategies(
                norm_speed=norm_speed,
                norm_acc=norm_acc,
                xi=self.xi,
                area_threshold=self.area_threshold,
            )
        elif self.hq_mode == "random":
            rng = np.random.default_rng(
                None if self.seed is None else int(self.seed) + int(self.times)
            )
            self.client_strategy = list(rng.choice(["PTQ", "QAT"], self.num_clients))
        else:
            self.client_strategy = [self.hq_mode.upper()] * self.num_clients
        for cid in range(self.num_clients):
            logging.info(
                f"FedHQ client {cid}: fit={fitted[cid]} "
                f"NormSpeed={norm_speed[cid]:.4f} NormAcc={norm_acc[cid]:.4f} "
                f"-> {self.client_strategy[cid]}"
            )
        # The deployed global model is the INT8 one the server sends down.
        self._commit_global(
            new_params=self.dequantize_int8(
                *self.quantize_int8(self.public_model_params)
            )
        )

    def package(self, client_id: int) -> "dict[str, Any]":
        package = super().package(client_id=client_id)
        q, scales = self.quantize_int8(package.pop("regular_model_params"))
        package.update(
            regular_model_params={},
            int8_params=q,
            int8_scales=scales,
            strategy=self.client_strategy[client_id],
            __wire__=("int8_params", "int8_scales"),
        )
        return package

    def aggregate_client_updates(
        self, packages: "OrderedDict[int, dict[str, Any]]"
    ) -> None:
        # Server dequantizes, aggregates, re-quantizes for the downlink.
        models = [
            self.dequantize_int8(package["int8_params"], package["int8_scales"])
            for package in packages.values()
        ]
        mean = self.mean_models(
            models=models, weights=[package["score"] for package in packages.values()]
        )
        new_params = OrderedDict(self.public_model_params)
        new_params.update(self.dequantize_int8(*self.quantize_int8(mean)))
        self._commit_global(new_params=new_params)


class FedHQ_Client(FedHQShared, tFL_Client):
    def set_parameters(self, package: "dict[str, Any]") -> None:
        super().set_parameters(package=package)
        self.model.load_state_dict(
            self.dequantize_int8(package["int8_params"], package["int8_scales"]),
            strict=False,
        )
        self.strategy = package["strategy"]

    def fit(self) -> None:
        if self.strategy == "PTQ":
            return super().fit()  # FP32 training; quantized only for upload
        self._set_worker_seed(seed=self._loader_seed(dataset_type="train"))
        loader = self.load_train_data()
        self.initialize_scheduler(steps_per_epoch=len(loader))
        model, device = self.model, self.device
        model.to(device)
        self._move_optimizer_state_to_param_devices(optimizer=self.optimizer)
        model.train()
        # Fake quantization between layers: every parameterized leaf's output.
        hooks = [
            module.register_forward_hook(
                lambda _m, _i, out: (
                    self.fake_quant(out)
                    if isinstance(out, torch.Tensor) and out.is_floating_point()
                    else out
                )
            )
            for module in model.modules()
            if not list(module.children())
            and any(True for _ in module.parameters(recurse=False))
        ]
        try:
            for _ in range(self.epochs):
                for batch_x, batch_y, x_mark, y_mark in loader:
                    self.optimizer.zero_grad(set_to_none=True)
                    batch_x = batch_x.to(device=device, dtype=torch.float32)
                    batch_y = batch_y.to(device=device, dtype=torch.float32)
                    x_mark = x_mark.to(device=device, dtype=torch.float32)
                    y_mark = y_mark.to(device=device, dtype=torch.float32)
                    weights = {
                        name: self.fake_quant(p)
                        for name, p in model.named_parameters()
                        if p.requires_grad
                    }
                    outputs = functional_call(
                        model, weights, (batch_x,), {"x_mark": x_mark, "y_mark": y_mark}
                    )
                    loss = self.loss(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()
                    self.step_scheduler_batch(
                        scheduler=self.scheduler, batch_data=batch_x
                    )
                self.step_scheduler_epoch(scheduler=self.scheduler)
        finally:
            for hook in hooks:
                hook.remove()
        if self.efficiency != "high":
            model.to("cpu")

    def package(self) -> "dict[str, Any]":
        package = super().package()
        q, scales = self.quantize_int8(package.pop("regular_model_params"))
        package.update(
            regular_model_params={},
            int8_params=q,
            int8_scales=scales,
            __wire__=("int8_params", "int8_scales", "score"),
        )
        return package
