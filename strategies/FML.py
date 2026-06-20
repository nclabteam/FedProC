import copy
from collections import OrderedDict

import torch
import torch.nn.functional as F

from losses import KLDivergence

from .pFL import pFL, pFL_Client


class FML(pFL):
    """FML: Federated Mutual Learning (Shen et al., 2020).

    Each client maintains a personal model (model_i) and a global model (model_g).
    Per round, model_g is synchronized via FedAvg; model_i is trained locally with
    mutual knowledge distillation between the two models:
      L_i = α * task_loss(model_i) + (1-α) * KL(log_σ(model_i) ‖ σ(model_g))
      L_g = β * task_loss(model_g) + (1-β) * KL(log_σ(model_g) ‖ σ(model_i))

    Default hyperparameters: α=0.9 (personal task weight), β=0.1 (global task weight).
    Reference: arXiv:2006.16765.
    """

    optional = {
        "alpha": 0.9,
        "beta": 0.1,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--alpha", type=float, default=None)
        parser.add_argument("--beta", type=float, default=None)

    def __init__(self, configs, times):
        super().__init__(configs=configs, times=times)
        init_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        for cid in range(self.num_clients):
            self.clients_personal_model_params[cid].update(
                {k: v.clone() for k, v in init_state.items()}
            )


class FML_Client(pFL_Client):
    def __init__(self, configs, times, device):
        super().__init__(configs, times, device)
        self.model_g = copy.deepcopy(self.model)
        obj = self._get_objective_function("optimizers", "Adam")
        self.optimizer_g = obj(params=self.model_g.parameters(), configs=self.configs)
        self.KL = KLDivergence()

    def set_parameters(self, package: dict) -> None:
        self.id = package["client_id"]
        self.current_iter = package["current_iter"]
        self._load_private(self.id)

        # regular_model_params carries the FedAvg'd model_g
        self.model_g.load_state_dict(package["regular_model_params"])

        # personal_model_params carries model_i state + optimizer_g state;
        # strict=False silently ignores "optimizer_g_state" key in model load.
        personal = package["personal_model_params"]
        self.model.load_state_dict(personal, strict=False)
        if "optimizer_g_state" in personal:
            self.optimizer_g.load_state_dict(personal["optimizer_g_state"])
            self._move_optimizer_state_to_param_devices(self.optimizer_g)

        if package["optimizer_state"]:
            self.optimizer.load_state_dict(package["optimizer_state"])
            self._move_optimizer_state_to_param_devices(self.optimizer)
        else:
            self.optimizer.load_state_dict(self.init_optimizer_state)
        if package["scheduler_state"]:
            self.scheduler.load_state_dict(package["scheduler_state"])
        else:
            self.scheduler.load_state_dict(self.init_scheduler_state)

    def fit(self) -> None:
        super().fit()
        if self.efficiency == "med":
            self.model_g.to("cpu")

    def package(self, train_time: float) -> dict:
        result = super().package(train_time)
        # regular_model_params = model_g (FedAvg'd by server each round)
        result["regular_model_params"] = OrderedDict(
            (k, v.detach().cpu().clone()) for k, v in self.model_g.state_dict().items()
        )
        # personal_model_params = model_i state + optimizer_g state (stored per-client)
        personal = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        personal["optimizer_g_state"] = self._optimizer_state_to_cpu(self.optimizer_g)
        result["personal_model_params"] = personal
        return result

    def train_one_epoch(self, dataloader, *args, offload_after=True, **kwargs):
        self.model.to(self.device)
        self.model_g.to(self.device)
        self._move_optimizer_state_to_param_devices(self.optimizer)
        self._move_optimizer_state_to_param_devices(self.optimizer_g)
        self.model.train()
        self.model_g.train()
        for batch_x, batch_y, x_mark, y_mark in dataloader:
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            x_mark = x_mark.to(self.device)
            y_mark = y_mark.to(self.device)
            output = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
            output_g = self.model_g(batch_x, x_mark=x_mark, y_mark=y_mark)
            loss = self.loss(output, batch_y) * self.alpha + self.KL(
                F.log_softmax(output, dim=1), F.softmax(output_g, dim=1)
            ) * (1 - self.alpha)
            loss_g = self.loss(output_g, batch_y) * self.beta + self.KL(
                F.log_softmax(output_g, dim=1), F.softmax(output, dim=1)
            ) * (1 - self.beta)
            self.optimizer.zero_grad()
            self.optimizer_g.zero_grad()
            loss.backward(retain_graph=True)
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            torch.nn.utils.clip_grad_norm_(self.model_g.parameters(), 10)
            self.optimizer.step()
            self.optimizer_g.step()
        self.scheduler.step()
        if offload_after:
            self.model.to("cpu")
            self.model_g.to("cpu")
