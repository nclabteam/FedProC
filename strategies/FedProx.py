import copy
from typing import Any, Dict

from .dFL import dFL, dFL_Client
from .tFL import tFL, tFL_Client


class FedProx(tFL):
    """FedProx: Federated Optimization with Proximal Term (Li et al., MLSys 2020).

    Adds a proximal regularization term (μ/2)||w - w_0||² to the client's local
    objective, applied via gradient accumulation: grad += μ * (w - w_0). This
    constrains local updates to stay close to the global model, improving stability
    under systems and statistical heterogeneity.

    Default μ explored in paper: {0.001, 0.01, 0.1, 1}. Reference: arXiv:1812.06127.
    """

    optional = {
        "mu": 0.01,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--mu", type=float, default=None)


class FedProx_Client(tFL_Client):
    def set_parameters(self, package: Dict[str, Any]) -> None:
        super().set_parameters(package)
        # Build ordered list of global param tensors (CPU) matching model.parameters() order
        self._global_params = [
            p.detach().cpu().clone()
            for p in package["regular_model_params"].values()
        ]

    def train_one_epoch(
        self,
        model,
        dataloader,
        optimizer,
        criterion,
        scheduler,
        device,
        offload_after=True,
    ):
        model.to(device)
        self._move_optimizer_state_to_param_devices(optimizer)
        global_params = [p.to(device) for p in self._global_params]
        model.train()
        for batch_x, batch_y, x_mark, y_mark in dataloader:
            optimizer.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            x_mark = x_mark.to(device)
            y_mark = y_mark.to(device)
            outputs = model(batch_x, x_mark=x_mark, y_mark=y_mark)
            loss = criterion(outputs, batch_y)
            loss.backward()
            for w, w_t in zip(model.parameters(), global_params):
                if w.requires_grad:
                    w.grad.data += self.mu * (w.data - w_t.data)
            optimizer.step()
        scheduler.step()
        if offload_after:
            model.to("cpu")
        del global_params


class DFedProx(dFL):
    """Decentralized FedProx: DFL aggregation plus local proximal training."""

    optional = {
        "mu": 0.01,
    }

    @classmethod
    def args_update(cls, parser):
        super().args_update(parser)
        parser.add_argument("--mu", type=float, default=None)


class DFedProx_Client(dFL_Client):
    def set_parameters(self, package):
        super().set_parameters(package)
        # Snapshot the gossip-aggregated model received from server as the prox center.
        self.snapshot = copy.deepcopy(self.model).to("cpu")

    def train_one_epoch(self, *args, **kwargs):
        return FedProx_Client.train_one_epoch(self, *args, **kwargs)
