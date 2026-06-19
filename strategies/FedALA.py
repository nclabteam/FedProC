import copy
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .pFL import pFL, pFL_Client


class FedALA(pFL):
    """Federated Adaptive Local Aggregation.

    Each client learns per-layer blending weights that mix the received global
    model with its own trained model before the next local training round.
    Server aggregation is standard FedAvg.

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

    def _apply_client_result(
        self, client, package: Optional[Dict[str, Any]]
    ) -> None:
        """Apply training package from :meth:`FedALA_Client.train` back to *client*.

        Extends the base implementation to restore the ALA-specific learnable
        blend weights and convergence state that are produced on the remote
        copy during Ray-parallel execution.

        Parameters
        ----------
        client : FedALA_Client
            Original client object to update.
        package : dict or None
            Return value of ``FedALA_Client.train()``.  Must contain all keys
            produced by ``tFL_Client.train()`` plus ``ala_weights`` and
            ``ala_start_phase``.
        """
        if package is None:
            return
        super()._apply_client_result(client, package)
        client.ala_weights = package["ala_weights"]
        client.ala_start_phase = package["ala_start_phase"]


class FedALA_Client(pFL_Client):
    """Client for FedALA.

    On each round:

    1. ``receive_from_server`` stores the incoming global model without loading
       it — the local model is preserved intact for the ALA blending step.
    2. ``train`` runs ALA (learns per-layer blend weights from a small random
       subset of local data), then performs standard gradient training on the
       blended model.

    The ALA state (``ala_weights``, ``ala_start_phase``) is included in the
    ``train`` return package so Ray-parallel execution can propagate it back to
    the original client.

    Attributes
    ----------
    _pending_global_model : nn.Module or None
        Global model received from the server, held for the ALA step inside
        ``train()``.
    ala_weights : list of Tensor or None
        Learnable per-parameter blend weights for the higher layers.
    ala_start_phase : bool
        ``True`` until ALA weight-learning converges for the first time.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_global_model: Optional[nn.Module] = None
        self.ala_weights: Optional[List[torch.Tensor]] = None
        self.ala_start_phase: bool = True

    def receive_from_server(self, data: Dict[str, Any]) -> None:
        """Store the global model for the ALA step; do not overwrite local model.

        Parameters
        ----------
        data : dict
            Server package containing ``"model"`` (the global nn.Module).
        """
        self._pending_global_model = data["model"]

    def train(self) -> Dict[str, Any]:
        """Run ALA blending followed by standard gradient training.

        Performs :meth:`_adaptive_local_aggregation` on the pending global
        model and the current local model, then delegates gradient training to
        :meth:`~tFL.tFL_Client.train` (the parent).

        Returns
        -------
        dict
            Package produced by ``tFL_Client.train()`` extended with
            ``ala_weights`` and ``ala_start_phase`` for parallel apply-back.
        """
        if self._pending_global_model is not None:
            self._adaptive_local_aggregation(
                global_model=self._pending_global_model,
                local_model=self.model,
            )
        package = super().train()
        package["ala_weights"] = self.ala_weights
        package["ala_start_phase"] = self.ala_start_phase
        return package

    def _adaptive_local_aggregation(
        self,
        global_model: nn.Module,
        local_model: nn.Module,
    ) -> None:
        """Learn per-layer blend weights and apply them to *local_model* in place.

        Updates ``self.ala_weights``, ``self.ala_start_phase``, and the
        higher-layer parameters of *local_model* in place.  Lower-layer
        parameters are overwritten with those of *global_model*.

        Parameters
        ----------
        global_model : nn.Module
            Global model received from the server.
        local_model : nn.Module
            Client's current local model (modified in place).
        """
        rand_loader = self.load_train_data(
            sample_ratio=self.sample_ratio, shuffle=False
        )

        params_g = list(global_model.parameters())
        params = list(local_model.parameters())

        if torch.sum(params_g[0] - params[0]) == 0:
            return

        for param, param_g in zip(
            params[: -self.layer_idx], params_g[: -self.layer_idx]
        ):
            param.data = param_g.data.clone()

        model_t = copy.deepcopy(local_model)
        params_t = list(model_t.parameters())
        params_p = params[-self.layer_idx :]
        params_gp = params_g[-self.layer_idx :]
        params_tp = params_t[-self.layer_idx :]

        for param in params_t[: -self.layer_idx]:
            param.requires_grad = False

        optimizer = torch.optim.SGD(params_tp, lr=0)

        if self.ala_weights is None:
            self.ala_weights = [torch.ones_like(param.data) for param in params_p]

        for param_t, param, param_g, weight in zip(
            params_tp, params_p, params_gp, self.ala_weights
        ):
            param_t.data = param + (param_g - param) * weight

        losses = []
        while True:
            for batch_x, batch_y, x_mark, y_mark in rand_loader:
                batch_x = batch_x.float()
                batch_y = batch_y.float()
                optimizer.zero_grad()
                output = model_t(batch_x, x_mark=x_mark, y_mark=y_mark)
                loss_value = self.loss(output, batch_y)
                loss_value.backward()

                for param_t, param, param_g, weight in zip(
                    params_tp, params_p, params_gp, self.ala_weights
                ):
                    weight.data = torch.clamp(
                        weight - self.eta * (param_t.grad * (param_g - param)), 0, 1
                    )
                    param_t.data = param + (param_g - param) * weight

            losses.append(loss_value.item())

            if not self.ala_start_phase:
                break
            self.logger.info(
                "ALA epochs: %03d | std: %.6f",
                len(losses),
                np.std(losses[-self.local_patience :]),
            )

            if (
                len(losses) > self.local_patience
                and np.std(losses[-self.local_patience :]) < self.threshold
            ):
                break

        self.ala_start_phase = False

        for param, param_t in zip(params_p, params_tp):
            param.data = param_t.data.clone()
