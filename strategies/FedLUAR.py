import math
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from collections.abc import Collection, Mapping
from typing import Any

import numpy as np
import torch

from .tFL import tFL, tFL_Client

_BYTES_PER_MB = 1024**2


class FedLUAR(tFL):
    """Layer-wise update aggregation with recycling."""

    optional = {
        "luar_num_recycle_layers": 1,
    }

    @classmethod
    def args_update(cls, parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "--luar_num_recycle_layers",
            type=int,
            default=None,
            help="Number of layers to recycle each round (delta)",
        )
        return parser

    @staticmethod
    def recyclable_layer_names(
        model: torch.nn.Module,
        public_names: Collection[str],
    ) -> list[str]:
        """Return trainable matrix parameters eligible for recycling."""
        return [
            name
            for name, parameter in model.named_parameters()
            if parameter.ndim > 1 and name in public_names
        ]

    @staticmethod
    def recycle_layers_mb(num_layers: int, delta: int) -> float:
        """Return the encoded recycled-layer IDs in MiB."""
        if delta <= 0 or num_layers <= 1:
            return 0.0
        return delta * math.ceil(math.log2(num_layers)) / 8 / _BYTES_PER_MB

    def __init__(self, configs: Namespace, times: int) -> None:
        super().__init__(configs=configs, times=times)
        if self.luar_num_recycle_layers < 0:
            raise ValueError("luar_num_recycle_layers must be nonnegative")
        # Saved params at round start: x_t (for computing deltas)
        self._luar_prev_params: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        # Combined delta from previous round: ˆΔ_{t-1}
        self._luar_prev_delta: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        # Importance scores s_{t,l}: {param_name: float}
        self._luar_scores: "OrderedDict[str, float]" = OrderedDict()
        self._luar_candidate_layers: list[str] = self.recyclable_layer_names(
            model=self.model,
            public_names=self.public_model_params,
        )
        # Recycling layers for current round R_t (names to SKIP)
        self._luar_recycle_layers: list[str] = []
        self._luar_first_round = True

    def package(self, client_id: int) -> dict:
        pkg = super().package(client_id=client_id)
        # Tell clients which layers to SKIP uploading
        pkg["luar_recycle_layers"] = self._luar_recycle_layers
        return pkg

    def _compute_send_mb(
        self,
        packages: Mapping[int, dict[str, Any]],
    ) -> tuple[dict[int, float], float]:
        uplink = {cid: self._uplink_sizes.get(cid, 0.0) for cid in packages}
        model_downlink = sum(
            self._downlink_sizes.get(cid, 0.0) for cid in self.selected_clients
        )
        rt_mb = self.recycle_layers_mb(
            num_layers=len(self._luar_candidate_layers),
            delta=len(self._luar_recycle_layers),
        )
        downlink = model_downlink + rt_mb * len(self.selected_clients)
        return uplink, downlink

    def select_clients(self) -> None:
        super().select_clients()
        self._update_layer_selection()

    def _compute_metric(self, agg_delta: "OrderedDict[str, torch.Tensor]") -> None:
        """Compute s_{t,l} = ||Δ_{t,l}|| / ||x_{t,l}|| for non-recycled layers."""
        for name in self._luar_candidate_layers:
            if name in self._luar_recycle_layers or self._luar_first_round:
                continue
            delta_norm = torch.norm(agg_delta[name].float()).item()
            weight_norm = torch.norm(self._luar_prev_params[name].float()).item()
            self._luar_scores[name] = delta_norm / (weight_norm + 1e-6)

    def _update_layer_selection(self) -> None:
        """Sample recycling layers."""
        param_names = self._luar_candidate_layers
        L = len(param_names)
        delta = min(self.luar_num_recycle_layers, max(0, L - 1)) if L > 0 else 0

        if self._luar_first_round or not self._luar_scores:
            self._luar_recycle_layers = []
            self._luar_first_round = False
            return

        scores = np.array(
            [self._luar_scores.get(name, 0.0) for name in param_names],
            dtype=float,
        )
        # Paper Eq. 2: p_l is proportional to 1 / s_l.
        inv_scores = 1.0 / np.maximum(scores, np.finfo(float).eps)
        probs = inv_scores / inv_scores.sum()

        # Paper Algorithm 1, pseudocode steps 7-8: sample R_(t+1).
        if delta > 0:
            seed = None if self.seed is None else self.seed + self.current_iter
            rng = np.random.default_rng(seed=seed)
            self._luar_recycle_layers = list(
                rng.choice(a=param_names, size=delta, replace=False, p=probs)
            )
        else:
            self._luar_recycle_layers = []

    def aggregate_client_updates(self, packages: "OrderedDict[int, dict]") -> None:
        """Aggregate with per-layer delta recycling."""
        if not packages:
            return

        # Preserve x_t for delta reconstruction.
        self._luar_prev_params = OrderedDict(
            (name, value.detach().clone())
            for name, value in self.public_model_params.items()
        )

        # Paper Algorithm 1, pseudocode step 3: simple (1/a) averaging:
        # u_t = (1/a) * sum_i(u_t^i), without sample-count weighting.
        client_models = [
            package.get("regular_model_params", {}) for package in packages.values()
        ]
        agg_params = OrderedDict()
        for name in self.public_model_params:
            if name in self._luar_recycle_layers:
                agg_params[name] = self._luar_prev_params[name].clone()
                continue
            agg_params[name] = torch.stack(
                [model[name] for model in client_models],
                dim=-1,
            ).mean(dim=-1)

        agg_delta = OrderedDict(
            (name, agg_params[name] - self._luar_prev_params[name])
            for name in self.public_model_params
        )

        # Paper Eq. 1 / Algorithm 1, pseudocode step 6: update s_(t,l).
        self._compute_metric(agg_delta=agg_delta)

        # Paper Algorithm 1, pseudocode steps 4-5: combine recycled and fresh deltas.
        combined_delta = OrderedDict()
        for name in self.public_model_params:
            if (
                not self._luar_first_round
                and name in self._luar_recycle_layers
                and name in self._luar_prev_delta
            ):
                combined_delta[name] = self._luar_prev_delta[name].clone()
            else:
                combined_delta[name] = agg_delta[name].clone()

        self._luar_prev_delta = OrderedDict(
            (name, value.clone()) for name, value in combined_delta.items()
        )

        # Paper Algorithm 2, pseudocode step 12: x_{t+1} = x_t + delta_hat_t.
        new_params = OrderedDict(
            (name, self._luar_prev_params[name] + combined_delta[name])
            for name in self.public_model_params
        )

        self._commit_global(new_params=new_params)


class FedLUAR_Client(tFL_Client):
    """FedLUAR Client: skips uploading recycled layers to reduce uplink."""

    def set_parameters(self, package: dict) -> None:
        super().set_parameters(package=package)
        self._luar_recycle_layers = package.get("luar_recycle_layers", [])

    def package(self) -> dict:
        result = super().package()
        # Paper Algorithm 1, pseudocode step 2: omit updates for R_t.
        if self._luar_recycle_layers:
            result["regular_model_params"] = OrderedDict(
                (name, value)
                for name, value in result["regular_model_params"].items()
                if name not in self._luar_recycle_layers
            )
        result["__wire__"] = ("regular_model_params",)
        return result
