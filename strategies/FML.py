import copy
from collections import OrderedDict
from typing import Any

import torch
import torch.nn.functional as F

from losses import KLDivergence

from .pFL import pFL, pFL_Client


class FML(pFL):
    """FML: Federated Mutual Learning (Shen et al., 2023)."""

    optional = {
        "alpha": 0.5,
        "beta": 0.5,
    }

    @classmethod
    def args_update(cls, parser: Any) -> None:
        parser.add_argument("--alpha", type=float, default=None)
        parser.add_argument("--beta", type=float, default=None)

    def __init__(self, configs: Any, times: Any) -> None:
        super().__init__(configs=configs, times=times)
        if not 0 <= self.alpha <= 1 or not 0 <= self.beta <= 1:
            raise ValueError("FML requires alpha and beta in [0, 1]")
        init_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        for cid in range(self.num_clients):
            self.clients_personal_model_params[cid].update(
                {k: v.clone() for k, v in init_state.items()}
            )

    def select_clients(self) -> None:
        self._select_all_clients()

    def aggregate_client_updates(self, packages: Any) -> None:
        self._commit_global(
            new_params=self.mean_models(
                models=[
                    package["regular_model_params"] for package in packages.values()
                ]
            )
        )


class FML_Client(pFL_Client):
    def __init__(self, configs: Any, times: Any, device: Any) -> None:
        super().__init__(configs=configs, times=times, device=device)
        self.model_g = copy.deepcopy(self.model)
        obj = self._get_objective_function(
            func_type="optimizers", func_name=self.configs.optimizer
        )
        self.optimizer_g = obj(params=self.model_g.parameters(), configs=self.configs)
        self.init_optimizer_g_state = copy.deepcopy(self.optimizer_g.state_dict())
        self.KL = KLDivergence()

    def set_parameters(self, package: dict) -> None:
        super().set_parameters(package=package)
        self.model_g.load_state_dict(package["regular_model_params"], strict=False)
        self.optimizer_g.load_state_dict(self.init_optimizer_g_state)

    def fit(self) -> None:
        super().fit()
        if self.efficiency == "med":
            self.model_g.to("cpu")

    def package(self) -> dict:
        result = super().package()
        result["regular_model_params"] = OrderedDict(
            (k, v.detach().cpu().clone()) for k, v in self.model_g.state_dict().items()
        )
        result["personal_model_params"] = {
            k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
        }
        return result

    def train_one_epoch(
        self, dataloader: Any, *args: Any, offload_after: Any = True, **kwargs: Any
    ) -> None:
        self.model.to(self.device)
        self.model_g.to(self.device)
        self._move_optimizer_state_to_param_devices(optimizer=self.optimizer)
        self._move_optimizer_state_to_param_devices(optimizer=self.optimizer_g)
        for personal_group, meme_group in zip(
            self.optimizer.param_groups, self.optimizer_g.param_groups
        ):
            meme_group["lr"] = personal_group["lr"]
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
                F.log_softmax(output, dim=1), F.softmax(output_g.detach(), dim=1)
            ) * (1 - self.alpha)
            loss_g = self.loss(output_g, batch_y) * self.beta + self.KL(
                F.log_softmax(output_g, dim=1), F.softmax(output.detach(), dim=1)
            ) * (1 - self.beta)
            self.optimizer.zero_grad(set_to_none=True)
            self.optimizer_g.zero_grad(set_to_none=True)
            loss.backward()
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            torch.nn.utils.clip_grad_norm_(self.model_g.parameters(), 10)
            self.optimizer.step()
            self.optimizer_g.step()
            self.step_scheduler_batch(
                scheduler=self.scheduler,
                batch_data=batch_x,
            )
        self.step_scheduler_epoch(scheduler=self.scheduler)
        if offload_after:
            self.model.to("cpu")
            self.model_g.to("cpu")
