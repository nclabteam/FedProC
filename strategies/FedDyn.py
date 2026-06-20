from collections import OrderedDict
from typing import Any, Dict, List

import torch

from .pFL import pFL, pFL_Client


class FedDyn(pFL):
    """FedDyn: Federated Learning Based on Dynamic Regularization (Acar et al., ICLR 2021).

    Client objective: L_k(x) - <∇L_k(x_k^{t-1}), x> + (α/2)||x - x^{t-1}||²
    Gradient correction: grad += -old_grad + α*(w - global_w)
    Dual update: old_grad -= α*(w_new - global_w)
    Server update: h += mean_clients - old_global; x_new = mean_clients - (1/N)*h

    Default α=0.1. Reference: arXiv:2111.04263. ICLR 2021.
    """

    optional = {
        "alpha": 0.1,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument(
            "--alpha",
            type=float,
            default=None,
            help="Alpha hyperparameter for FedDyn (regularization strength)",
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_state: List[torch.Tensor] = [
            torch.zeros_like(p).cpu() for p in self.public_model_params.values()
        ]
        zero_grads = [torch.zeros_like(p).cpu() for p in self.model.parameters()]
        for cid in range(self.num_clients):
            self.clients_personal_model_params[cid]["old_grad"] = [
                t.clone() for t in zero_grads
            ]

    def package(self, client_id: int) -> Dict[str, Any]:
        pkg = super().package(client_id)
        pkg["global_params_list"] = [
            p.detach().cpu().clone() for p in self.public_model_params.values()
        ]
        return pkg

    def aggregate_client_updates(self, packages: "OrderedDict[int, dict]") -> None:
        # Snapshot current global params before updating
        old_global: List[torch.Tensor] = [
            p.detach().cpu().clone() for p in self.public_model_params.values()
        ]

        # Sample-weighted mean of client regular params
        scores = [p["score"] for p in packages.values()]
        total = float(sum(scores))
        weights = [s / total for s in scores]

        param_names = list(self.public_model_params.keys())
        mean_params: List[torch.Tensor] = []
        for name in param_names:
            stacked = torch.stack(
                [p["regular_model_params"][name].cpu() for p in packages.values()],
                dim=-1,
            )
            w = torch.tensor(weights, dtype=stacked.dtype)
            mean_params.append(torch.sum(stacked * w, dim=-1))

        # server_state[i] += mean_params[i] - old_global[i]
        for i in range(len(self.server_state)):
            self.server_state[i] = self.server_state[i] + mean_params[i] - old_global[i]

        # new_global[i] = mean_params[i] - (1/N) * server_state[i]
        N = self.num_clients
        new_params = OrderedDict()
        for i, name in enumerate(param_names):
            new_params[name] = mean_params[i] - (1.0 / N) * self.server_state[i]

        self._commit_global(new_params)


class FedDyn_Client(pFL_Client):
    """Client for FedDyn.

    Per-client persistent state in personal_model_params:
        "old_grad" — list of CPU tensors (dual variable, one per model param).
    """

    def set_parameters(self, package: Dict[str, Any]) -> None:
        super().set_parameters(package)
        self._old_grad: List[torch.Tensor] = [
            t.detach().cpu().clone()
            for t in package["personal_model_params"]["old_grad"]
        ]
        self._global_params_list: List[torch.Tensor] = [
            p.detach().cpu().clone() for p in package["global_params_list"]
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
        global_params = [p.to(device) for p in self._global_params_list]
        old_grad = [g.to(device) for g in self._old_grad]
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
            # FedDyn gradient correction: -old_grad + alpha*(w - global_w)
            for w, gp, og in zip(model.parameters(), global_params, old_grad):
                if w.requires_grad and w.grad is not None:
                    w.grad.data += -og + self.alpha * (w.data - gp.data)
            optimizer.step()
        scheduler.step()
        if offload_after:
            model.to("cpu")
        del global_params, old_grad

    def package(self, train_time: float) -> Dict[str, Any]:
        out = super().package(train_time)
        # Dual-variable update: old_grad_i -= alpha * (w_i_new - global_w_i)
        updated_old_grad = [
            og - self.alpha * (w.detach().cpu() - gp)
            for og, w, gp in zip(
                self._old_grad,
                self.model.parameters(),
                self._global_params_list,
            )
        ]
        out["personal_model_params"]["old_grad"] = updated_old_grad
        return out
