import copy

import numpy as np
import torch
import torch.nn as nn

from .base import Client, Server

# Default options
optional = {
    "eta": 1.0,
    "sample_ratio": 0.8,
    "layer_idx": 2,
    "threshold": 0.1,
    "local_patience": 10,
}

compulsory = {
    "save_local_model": True,
}


# Argument parser update function
def args_update(parser):
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--sample_ratio", type=float, default=None)
    parser.add_argument("--layer_idx", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--local_patience", type=int, default=None)


class FedALA(Server):
    pass


class FedALA_Client(Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = None  # Learnable local aggregation weights
        self.start_phase = True

    def receive_from_server(self, data):
        super().update_model_params(
            old=self.model,
            new=self.adaptive_local_aggregation(
                global_model=data["model"],
                local_model=self.model,
            ),
        )

    def adaptive_local_aggregation(
        self,
        global_model: nn.Module,
        local_model: nn.Module,
    ) -> None:
        """
        Performs Adaptive Local Aggregation (ALA) with partial local training data
        and preserves updates in the lower layers.

        Args:
            global_model (nn.Module): The received global model.
            local_model (nn.Module): The trained local model.
        """
        # Generate a DataLoader with shuffled data
        rand_loader = self.load_train_data(
            sample_ratio=self.sample_ratio, shuffle=False
        )

        # Obtain parameter references
        params_g = list(global_model.parameters())
        params = list(local_model.parameters())

        # Deactivate ALA during the first communication iteration
        if torch.sum(params_g[0] - params[0]) == 0:
            return local_model

        # Preserve updates in lower layers
        for param, param_g in zip(
            params[: -self.layer_idx], params_g[: -self.layer_idx]
        ):
            param.data = param_g.data.clone()

        # Temporary local model for weight learning
        model_t = copy.deepcopy(local_model)
        params_t = list(model_t.parameters())
        params_p, params_gp, params_tp = (
            params[-self.layer_idx :],
            params_g[-self.layer_idx :],
            params_t[-self.layer_idx :],
        )

        # Freeze lower layers to reduce computation
        for param in params_t[: -self.layer_idx]:
            param.requires_grad = False

        # Initialize optimizer with lr=0 since optimizer.step() is not used
        optimizer = torch.optim.SGD(params_tp, lr=0)

        # Initialize weights if not already done
        if self.weights is None:
            self.weights = [torch.ones_like(param.data) for param in params_p]

        # Initialize higher layers in the temp model
        for param_t, param, param_g, weight in zip(
            params_tp, params_p, params_gp, self.weights
        ):
            param_t.data = param + (param_g - param) * weight

        # Weight learning
        losses = []
        while True:
            for batch_x, batch_y in rand_loader:
                batch_x = batch_x.float()
                batch_y = batch_y.float()
                optimizer.zero_grad()
                output = model_t(batch_x)
                loss_value = self.loss(output, batch_y)  # Local objective
                loss_value.backward()

                # Update weights and temporary model
                for param_t, param, param_g, weight in zip(
                    params_tp, params_p, params_gp, self.weights
                ):
                    weight.data = torch.clamp(
                        weight - self.eta * (param_t.grad * (param_g - param)), 0, 1
                    )
                    param_t.data = param + (param_g - param) * weight

            losses.append(loss_value.item())

            # Stop if not in the start phase
            if not self.start_phase:
                break
            self.logger.info(
                f"Std: {np.std(losses[-self.local_patience:]):.6f} | ALA epochs: {len(losses):03d}"
            )

            # Check for convergence
            if (
                len(losses) > self.local_patience
                and np.std(losses[-self.local_patience :]) < self.threshold
            ):
                break

        self.start_phase = False

        # Update local model with temp model
        for param, param_t in zip(params_p, params_tp):
            param.data = param_t.data.clone()

        return local_model
