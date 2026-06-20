import copy
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .pFL import pFL, pFL_Client

_logger = logging.getLogger(__name__)


class FedALA(pFL):
    """Federated Adaptive Local Aggregation.

    Each client learns per-layer blending weights that mix the received global
    model with its own previously trained model before each round's local
    training.  Server aggregation is standard FedAvg (inherited from pFL/tFL).

    References
    ----------
    Yutao Huang et al., "FedALA: Adaptive Local Aggregation for Personalized
    Federated Learning," AAAI 2023.
    """

    optional = {
        "eta": 1.0,
        "sample_ratio": 0.8,
        "layer_idx": 2,
        "threshold": 0.1,
        "local_patience": 10,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--eta", type=float, default=None)
        parser.add_argument("--sample_ratio", type=float, default=None)
        parser.add_argument("--layer_idx", type=int, default=None)
        parser.add_argument("--threshold", type=float, default=None)
        parser.add_argument("--local_patience", type=int, default=None)


class FedALA_Client(pFL_Client):
    """Client for FedALA.

    Per-client persistent state in personal_model_params:
        "ala_weights"       — list of CPU tensors (per-param blend weights for
                              the last ``layer_idx`` parameter groups), or None.
        "ala_start_phase"   — bool; True until ALA weights converge for the
                              first time, then False (one-pass mode).
        "prev_local_params" — list of CPU tensors (all model params after the
                              previous round's training), or None on first round.
    """

    def set_parameters(self, package: Dict[str, Any]) -> None:
        super().set_parameters(package)  # loads global model into self.model
        pm = package["personal_model_params"]
        self._ala_weights: Optional[List[torch.Tensor]] = pm.get("ala_weights", None)
        self._ala_start_phase: bool = pm.get("ala_start_phase", True)
        self._prev_local_params: Optional[List[torch.Tensor]] = pm.get(
            "prev_local_params", None
        )

    def fit(self) -> None:
        # Run ALA interpolation before local training if we have a previous
        # local model to blend with.
        if self._prev_local_params is not None:
            self._run_ala()
        super().fit()

    def _run_ala(self) -> None:
        """Adaptive Local Aggregation: interpolate self.model in-place.

        After super().set_parameters() the model holds global params.
        This method:
          1. Copies global params into the lower layers (already done by
             set_parameters; no-op here).
          2. Learns per-param blend weights for the higher layers via a
             small-batch gradient descent on the local objective.
          3. Applies the learned weights to set the higher layers of
             self.model to the interpolated values.
        """
        # Global params: what set_parameters already loaded into self.model
        global_params = [p.detach().cpu().clone() for p in self.model.parameters()]
        prev_local = [t.clone() for t in self._prev_local_params]

        # Skip if global and previous local are identical (e.g. first real round)
        if torch.sum(global_params[0] - prev_local[0]) == 0:
            return

        # Higher-layer slices used for weight learning / interpolation
        params_p = prev_local[-self.layer_idx:]     # prev local, higher layers
        params_gp = global_params[-self.layer_idx:] # global,     higher layers

        if self._ala_weights is None:
            self._ala_weights = [torch.ones_like(p) for p in params_p]

        # model_t: deep copy of self.model (global params everywhere).
        # Lower layers stay as global; higher layers are set to interpolated init.
        model_t = copy.deepcopy(self.model)
        params_t = list(model_t.parameters())
        params_tp = params_t[-self.layer_idx:]

        for param in params_t[:-self.layer_idx]:
            param.requires_grad = False

        with torch.no_grad():
            for pt, pp, pg, w in zip(params_tp, params_p, params_gp, self._ala_weights):
                pt.data = pp + (pg - pp) * w

        # SGD optimizer with lr=0: we update weights manually, not via step()
        optimizer = torch.optim.SGD(params_tp, lr=0)

        rand_loader = self.load_data(
            file=self.train_file,
            sample_ratio=self.sample_ratio,
            batch_size=self.batch_size,
            shuffle=False,
            scaler=self.scaler,
            seed=self._loader_seed("train"),
        )

        losses: List[float] = []
        loss_value = torch.tensor(0.0)
        while True:
            for batch_x, batch_y, x_mark, y_mark in rand_loader:
                batch_x = batch_x.float()
                batch_y = batch_y.float()
                optimizer.zero_grad()
                output = model_t(batch_x, x_mark=x_mark, y_mark=y_mark)
                loss_value = self.loss(output, batch_y)
                loss_value.backward()
                with torch.no_grad():
                    for pt, pp, pg, w in zip(
                        params_tp, params_p, params_gp, self._ala_weights
                    ):
                        w.data = torch.clamp(
                            w - self.eta * (pt.grad * (pg - pp)), 0, 1
                        )
                        pt.data = pp + (pg - pp) * w

            losses.append(loss_value.item())

            if not self._ala_start_phase:
                break
            _logger.info(
                "ALA epochs: %03d | std: %.6f",
                len(losses),
                np.std(losses[-self.local_patience:]),
            )
            if (
                len(losses) > self.local_patience
                and np.std(losses[-self.local_patience:]) < self.threshold
            ):
                break

        self._ala_start_phase = False

        # Write learned higher-layer params back to self.model
        with torch.no_grad():
            model_params = list(self.model.parameters())
            for mp, pt in zip(model_params[-self.layer_idx:], params_tp):
                mp.data.copy_(pt.data)

        del model_t

    def package(self, train_time: float) -> Dict[str, Any]:
        out = super().package(train_time)
        # Persist ALA state and current trained params for the next round
        out["personal_model_params"]["prev_local_params"] = [
            p.detach().cpu().clone() for p in self.model.parameters()
        ]
        out["personal_model_params"]["ala_weights"] = self._ala_weights
        out["personal_model_params"]["ala_start_phase"] = self._ala_start_phase
        return out
