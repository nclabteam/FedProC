import math
from collections import OrderedDict

import numpy as np
import torch

from .tFL import tFL, tFL_Client

_BYTES_PER_MB = 1024 ** 2


def _recycle_layers_mb(num_layers: int, delta: int) -> float:
    """Paper (line 322): R_t (the delta recycled-layer IDs) is transmitted
    downlink alongside the model. Encoded as delta integers, each needing
    ceil(log2(num_layers)) bits to index one of num_layers layers.
    """
    if delta <= 0 or num_layers <= 1:
        return 0.0
    bits_per_index = math.ceil(math.log2(num_layers))
    return delta * bits_per_index / 8 / _BYTES_PER_MB


class FedLUAR(tFL):
    """FedLUAR: Layer-wise Update Aggregation with Recycling (paper 10021).

    Per-round protocol:
      1. Server selects delta = luar_num_recycle_layers layers for recycling
         via random sampling with probability p_l ∝ 1/s_l.
      2. Clients train locally, compute delta = params_after - params_before,
         and upload only non-recycled layer deltas (full params in FedProC).
      3. Server aggregates new deltas for non-recycled layers, reuses previous
         round's combined delta for recycled layers.
      4. Server applies: x_{t+1} = x_t + combined_delta.
      5. Metric s_{t,l} = ||Δ_{t,l}|| / ||x_{t,l}|| recomputed fresh each round
         for non-recycled layers (Eq. 1); recycled layers keep old s.

    Reference: arXiv:2503.11146 (NeurIPS 2025).
    """

    optional = {
        "luar_num_recycle_layers": 1,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--luar_num_recycle_layers", type=int, default=None,
            help="Number of layers to recycle each round (delta)")
        return parser

    def __init__(self, configs, times):
        super().__init__(configs, times)
        # Saved params at round start: x_t (for computing deltas)
        self._luar_prev_params: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        # Combined delta from previous round: ˆΔ_{t-1}
        self._luar_prev_delta: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        # Importance scores s_{t,l}: {param_name: float}
        self._luar_scores: "OrderedDict[str, float]" = OrderedDict()
        # Recycling layers for current round R_t (names to SKIP)
        self._luar_recycle_layers: list = []
        self._luar_first_round = True

    def package(self, client_id: int) -> dict:
        pkg = super().package(client_id)
        # Tell clients which layers to SKIP uploading
        pkg["luar_recycle_layers"] = self._luar_recycle_layers
        return pkg

    def _compute_send_mb(self, packages) -> tuple:
        uplink = {cid: self._uplink_sizes.get(cid, 0.0) for cid in packages}
        model_downlink = sum(
            self._downlink_sizes.get(cid, 0.0) for cid in self.selected_clients
        )
        rt_mb = _recycle_layers_mb(
            len(self.public_model_params), len(self._luar_recycle_layers)
        )
        downlink = model_downlink + rt_mb * len(self.selected_clients)
        return uplink, downlink

    def select_clients(self) -> None:
        super().select_clients()
        self._update_layer_selection()

    def _compute_metric(self, agg_delta: "OrderedDict[str, torch.Tensor]") -> None:
        """Compute s_{t,l} = ||Δ_{t,l}|| / ||x_{t,l}|| for non-recycled layers."""
        for name in self.public_model_params:
            if name in self._luar_recycle_layers or self._luar_first_round:
                continue
            delta_norm = torch.norm(agg_delta[name].float()).item()
            weight_norm = torch.norm(self._luar_prev_params[name].float()).item()
            self._luar_scores[name] = delta_norm / weight_norm if weight_norm > 0 else 0.0

    def _update_layer_selection(self) -> None:
        """Sample δ recycling layers using p_l ∝ 1/s_l (Eq. 2)."""
        param_names = list(self.public_model_params.keys())
        L = len(param_names)
        delta = min(self.luar_num_recycle_layers, max(0, L - 1)) if L > 0 else 0

        if self._luar_first_round or not self._luar_scores:
            self._luar_recycle_layers = []
            self._luar_first_round = False
            return

        scores = np.array([self._luar_scores.get(n, 1e-10) for n in param_names])
        inv_scores = 1.0 / scores
        probs = inv_scores / inv_scores.sum()

        if delta > 0:
            rng = np.random.default_rng(self.seed + self.current_iter)
            self._luar_recycle_layers = list(
                rng.choice(param_names, delta, replace=False, p=probs)
            )
        else:
            self._luar_recycle_layers = []

    def aggregate_client_updates(self, packages: "OrderedDict[int, dict]") -> None:
        """Aggregate using per-layer delta recycling (Alg. 1)."""
        if not packages:
            return

        # 1. Save x_t (params at round start)
        self._luar_prev_params = OrderedDict(
            (k, v.data.clone()) for k, v in self.public_model_params.items()
        )

        # 2. Simple (1/a) averaging of returned params (Alg. 1, line 2:
        #    u_t = (1/a) * sum(u_t^i) -- NOT sample-count weighted)
        a = len(packages)
        agg_params = OrderedDict()
        for name in self.public_model_params:
            available = all(
                name in p.get("regular_model_params", {}) for p in packages.values()
            )
            if available:
                stacked = torch.stack(
                    [p["regular_model_params"][name] for p in packages.values()], dim=-1
                )
                agg_params[name] = stacked.sum(dim=-1) / a
            else:
                agg_params[name] = self._luar_prev_params[name].data.clone()

        # 3. Compute fresh aggregated delta for non-recycled layers
        #    Δ_{t,l} = agg_params[l] - x_{t,l}
        agg_delta = OrderedDict()
        for name in self.public_model_params:
            agg_delta[name] = agg_params[name] - self._luar_prev_params[name]

        # 4. Update metric s_{t,l} (Eq. 1) using fresh delta for non-recycled
        self._compute_metric(agg_delta)

        # 5. Build combined delta ˆΔ_t
        combined_delta = OrderedDict()
        for name in self.public_model_params:
            if not self._luar_first_round and name in self._luar_recycle_layers and name in self._luar_prev_delta:
                # Recycle previous round's delta r_t = ˆΔ_{t-1,l}
                combined_delta[name] = self._luar_prev_delta[name].data.clone()
            else:
                # Use fresh aggregated delta u_t = Δ_{t,l}
                combined_delta[name] = agg_delta[name].data.clone()

        # 6. Store ˆΔ_t for next round
        self._luar_prev_delta = OrderedDict(
            (k, v.data.clone()) for k, v in combined_delta.items()
        )

        # 7. Apply: x_{t+1} = x_t + ˆΔ_t (Alg. 2 line 12)
        new_params = OrderedDict()
        for name in self.public_model_params:
            new_params[name] = self._luar_prev_params[name] + combined_delta[name]

        self._commit_global(new_params)

    def _commit_global(self, new_params) -> None:
        self.public_model_params = OrderedDict(new_params)
        self.model.load_state_dict(self.public_model_params, strict=False)


class FedLUAR_Client(tFL_Client):
    """FedLUAR Client: skips uploading recycled layers to reduce uplink."""

    def set_parameters(self, package: dict) -> None:
        super().set_parameters(package)
        self._luar_recycle_layers = package.get("luar_recycle_layers", [])

    def package(self) -> dict:
        result = super().package()
        # Upload all layers EXCEPT recycled ones
        if self._luar_recycle_layers:
            filtered = OrderedDict()
            for name in result["regular_model_params"]:
                if name not in self._luar_recycle_layers:
                    filtered[name] = result["regular_model_params"][name]
            result["regular_model_params"] = filtered
        result["__wire__"] = ("regular_model_params", "score")
        return result
