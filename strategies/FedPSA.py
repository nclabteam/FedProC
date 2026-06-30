# -*- coding: utf-8 -*-
"""FedPSA: Parameter Sensitivity-based Asynchronous Federated Learning.

Behavioral staleness metric: cosine similarity of parameter-sensitivity sketches.
Training thermometer: queue of update magnitudes adjusts softmax temperature.

Paper: arxiv.org/abs/2602.15337 (ID 08958)

Algorithm (paper Alg. 1):
  Client: LocalUpdate → compute sensitivity s_i on shared calib batch D_b →
          sketch s̃_i = R s_i → upload (Δw_i, s̃_i, m_i)
  Server: on buffer full → compute s̃_g → κ_i = cos(s̃_i, s̃_g) →
          Temp from thermometer queue → Weight_i = softmax(κ_i / Temp) →
          w_g += Σ Weight_i Δw_i
"""
import math
from collections import deque
from typing import Any, Dict

import torch
import torch.nn.functional as F

from .aFL import aFL
from .tFL import tFL_Client


def _sensitivity_sketch(model: torch.nn.Module, calib_x: torch.Tensor,
                        calib_y: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Compute sensitivity vector and project to sketch space.

    Sensitivity (eq 9): s_j = |g_j θ_j - 0.5 F_jj θ_j²|
    Fisher diagonal approximated as g_j² (single-batch squared gradient).
    Sketch: s̃ = R s  (R ∈ R^{k×d})
    Returns k-dim CPU tensor.
    """
    device = next(model.parameters()).device
    x = calib_x.to(device)
    y = calib_y.to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    model.zero_grad()
    try:
        out = model(x)
    except TypeError:
        # Some models need extra args; fall back to zeros sensitivity
        return torch.zeros(R.shape[0])
    loss = F.mse_loss(out, y)
    loss.backward()

    grad = torch.cat([
        p.grad.detach().flatten() if p.grad is not None
        else torch.zeros(p.numel(), device=device)
        for p in params
    ])
    theta = torch.cat([p.detach().flatten() for p in params])
    fisher = grad.pow(2)  # diagonal Fisher approx via squared batch gradient

    s = (grad * theta - 0.5 * fisher * theta.pow(2)).abs()
    model.zero_grad()
    return (R.to(device) @ s).detach().cpu()


class FedPSA(aFL):
    """FedPSA server.

    Inherits the buffered-async training loop from aFL and overrides
    aggregate_client_updates with the sensitivity-based softmax weighting.
    """

    optional = {
        "buffer_size": 10,
        "queue_len": 20,
        "sketch_dim": 256,
        "gamma": 1.0,
        "delta": 0.1,
        "calib_size": 32,
    }

    def __init__(self, configs, times) -> None:
        super().__init__(configs, times)
        d = sum(p.numel() for p in self.model.parameters())
        # Fixed projection matrix R ∈ R^{k×d}, shared via seed
        self._R_seed = 0
        gen = torch.Generator().manual_seed(self._R_seed)
        self._R = torch.randn(self.sketch_dim, d, generator=gen) / math.sqrt(self.sketch_dim)
        # Synthetic calibration batch (channel-independent: batch × input_len)
        gen2 = torch.Generator().manual_seed(1)
        self._calib_x = torch.randn(self.calib_size, self.input_len, generator=gen2)
        self._calib_y = torch.randn(self.calib_size, self.output_len, generator=gen2)
        # Thermometer
        self._thermo: deque = deque(maxlen=self.queue_len)
        self._M0: float = None

    def package(self, client_id: int) -> dict:
        pkg = super().package(client_id)
        pkg["_psa_R_seed"] = self._R_seed
        pkg["_psa_sketch_dim"] = self.sketch_dim
        pkg["_psa_calib_x"] = self._calib_x
        pkg["_psa_calib_y"] = self._calib_y
        return pkg

    def aggregate_client_updates(self, packages) -> None:
        # --- thermometer update ---
        for pkg in packages.values():
            self._thermo.append(pkg["_psa_m_i"])

        M_cur = sum(self._thermo) / len(self._thermo)
        if self._M0 is None and len(self._thermo) >= self.queue_len:
            self._M0 = M_cur

        temp = None if self._M0 is None else (M_cur / self._M0) * self.gamma + self.delta

        # --- server sensitivity sketch ---
        s_g = _sensitivity_sketch(self.model, self._calib_x, self._calib_y, self._R)
        s_g_norm = s_g.norm().clamp(min=1e-8)

        # --- κ_i = cos(s̃_i, s̃_g) ---
        kappas = [
            (torch.dot(pkg["_psa_s_tilde"], s_g) / (pkg["_psa_s_tilde"].norm().clamp(min=1e-8) * s_g_norm)).item()
            for pkg in packages.values()
        ]

        # --- softmax weights ---
        if temp is None:
            weights = [1.0 / len(packages)] * len(packages)
        else:
            weights = F.softmax(torch.tensor(kappas) / temp, dim=0).tolist()

        # --- w_g^new = w_g^old + Σ weight_i Δw_i  (eq 13) ---
        new_params = {k: v.clone() for k, v in self.public_model_params.items()}
        for (_, pkg), w in zip(packages.items(), weights):
            for k, dv in pkg["_psa_delta"].items():
                new_params[k] = new_params[k] + w * dv
        self._commit_global(new_params)

        self.logger.info(
            "FedPSA agg: temp=%.4f  κ=[%s]",
            temp if temp is not None else -1.0,
            " ".join(f"{k:.3f}" for k in kappas),
        )


class FedPSA_Client(tFL_Client):
    _R: torch.Tensor = None
    _calib_x: torch.Tensor = None
    _calib_y: torch.Tensor = None
    _w0: Dict[str, torch.Tensor] = None

    def set_parameters(self, package: dict) -> None:
        super().set_parameters(package)
        # Record global model before local training for delta computation
        self._w0 = {k: v.detach().cpu().clone() for k, v in self.model.named_parameters()}
        # Build R once from seed (avoids sending k×d matrix every round)
        if self._R is None:
            seed = package["_psa_R_seed"]
            k = package["_psa_sketch_dim"]
            d = sum(p.numel() for p in self.model.parameters())
            gen = torch.Generator().manual_seed(seed)
            self._R = torch.randn(k, d, generator=gen) / math.sqrt(k)
        self._calib_x = package["_psa_calib_x"]
        self._calib_y = package["_psa_calib_y"]

    def package(self) -> Dict[str, Any]:
        result = super().package()
        # Δw_i = w_after - w_before
        delta = {k: v.detach().cpu() - self._w0[k] for k, v in self.model.named_parameters()}
        flat_delta = torch.cat([v.flatten() for v in delta.values()])
        result["_psa_delta"] = delta
        result["_psa_m_i"] = float(flat_delta.pow(2).sum())
        result["_psa_s_tilde"] = _sensitivity_sketch(
            self.model, self._calib_x, self._calib_y, self._R
        )
        return result
