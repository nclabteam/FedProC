import copy
from typing import Any, Dict

import numpy as np
import torch

from .pFL import pFL, pFL_Client


class APFL(pFL):
    """APFL: Adaptive Personalized Federated Learning (Deng et al., 2020).

    Each client maintains a global model w (aggregated via FedAvg) and a
    personalized model v_i.  Per round, both models are trained simultaneously
    and their outputs mixed with a per-client adaptive coefficient α_i.
    α_i is updated via gradient descent on the mixed objective each batch.

    Default α=0.5 (initial mixing coefficient). Reference: arXiv:2003.13461.
    """

    optional = {
        "alpha": 0.5,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--alpha", type=float, default=None)

    def __init__(self, configs, times):
        super().__init__(configs=configs, times=times)
        init_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        for cid in range(self.num_clients):
            self.clients_personal_model_params[cid]["model_per"] = {
                k: v.clone() for k, v in init_state.items()
            }
            self.clients_personal_model_params[cid]["alpha"] = self.alpha


class APFL_Client(pFL_Client):
    """Client for APFL.

    Trains global model w (sent to server) and personalized model v_i (kept local)
    simultaneously, updating mixing coefficient α_i adaptively each batch.
    Per-client persistent state: "model_per" (state dict of v_i), "alpha" (α_i).
    """

    def __init__(self, configs, times, device):
        super().__init__(configs, times, device)
        self.model_per = copy.deepcopy(self.model)

    def set_parameters(self, package: Dict[str, Any]) -> None:
        super().set_parameters(package)
        self.model_per.load_state_dict(package["personal_model_params"]["model_per"])
        self.alpha = package["personal_model_params"]["alpha"]

    def fit(self) -> None:
        self._set_worker_seed(self._loader_seed("train"))
        loader = self.load_train_data()
        optim_p = torch.optim.SGD(self.model_per.parameters(), lr=self.learning_rate)

        self.model.to(self.device)
        self._move_optimizer_state_to_param_devices(self.optimizer)
        self.model_per.to(self.device)
        self.model.train()
        self.model_per.train()

        for _ in range(self.epochs):
            for batch_x, batch_y, x_mark, y_mark in loader:
                batch_x = batch_x.to(device=self.device, dtype=torch.float32)
                batch_y = batch_y.to(device=self.device, dtype=torch.float32)
                x_mark = x_mark.to(device=self.device, dtype=torch.float32)
                y_mark = y_mark.to(device=self.device, dtype=torch.float32)

                # Train global model
                self.optimizer.zero_grad()
                out_g = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
                loss_g = self.loss(out_g, batch_y)
                loss_g.backward()
                self.optimizer.step()

                # Train personalized model
                optim_p.zero_grad()
                out_p = self.model_per(batch_x, x_mark=x_mark, y_mark=y_mark)
                loss_p = self.loss(out_p, batch_y)
                loss_p.backward()
                optim_p.step()

                # Adaptive alpha update
                self._update_alpha()

            self.scheduler.step()

        # Mix: v_i = (1-α)*w + α*v_i
        for lp, gp in zip(self.model_per.parameters(), self.model.parameters()):
            lp.data = (1.0 - self.alpha) * gp.data + self.alpha * lp.data

        self.model.to("cpu")
        self.model_per.to("cpu")

    def _update_alpha(self) -> None:
        grad_alpha = 0.0
        for gp, pp in zip(self.model.parameters(), self.model_per.parameters()):
            if pp.grad is None or gp.grad is None:
                continue
            dif = (pp.data - gp.data).view(-1)
            grad = (self.alpha * pp.grad.data + (1.0 - self.alpha) * gp.grad.data).view(
                -1
            )
            grad_alpha += float(dif.dot(grad))
        self.alpha = float(
            np.clip(self.alpha - self.learning_rate * grad_alpha, 0.0, 1.0)
        )

    def package(self, train_time: float) -> Dict[str, Any]:
        result = super().package(train_time)
        result["personal_model_params"]["model_per"] = {
            k: v.detach().cpu().clone()
            for k, v in self.model_per.state_dict().items()
        }
        result["personal_model_params"]["alpha"] = self.alpha
        return result

    def evaluate_personalized(
        self, client_id, global_params, personal_params, dataset_type, current_iter
    ) -> float:
        self.id = client_id
        self.current_iter = current_iter
        self._load_private(client_id)
        self.model.load_state_dict(global_params, strict=False)
        self.model.load_state_dict(personal_params["model_per"], strict=False)
        loader = (
            self.load_test_data()
            if dataset_type == "test"
            else self.load_train_data()
        )
        losses = self.calculate_loss(
            model=self.model,
            dataloader=loader,
            criterion=self.loss,
            device=self.device,
            offload_after=self.efficiency != "high",
        )
        return float(np.mean(losses))
