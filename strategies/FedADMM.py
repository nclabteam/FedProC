import copy
from collections import OrderedDict
from typing import Any, Dict

import torch

from .tFL import tFL, tFL_Client


class FedADMMShared:
    """Paper-level operations shared by the FedADMM server and client."""

    @staticmethod
    def augmented_delta(
        previous_w: Any, previous_y: Any, current_w: Any, current_y: Any, rho: Any
    ) -> Any:
        if rho <= 0:
            raise ValueError("rho must be positive")
        return OrderedDict(
            (
                name,
                current_w[name]
                + current_y[name] / rho
                - previous_w[name]
                - previous_y[name] / rho,
            )
            for name in current_w
        )


class FedADMM(FedADMMShared, tFL):
    """FedADMM (Gong et al., ICDE 2022)."""

    optional = {
        "rho": 0.01,
        "server_learning_rate": 1.0,
        "server_learning_rate_2": 0.5,
        "target_round": 60,
        "fixed": 0,
    }

    @classmethod
    def args_update(cls, parser: Any) -> None:
        parser.add_argument("--rho", type=float, default=None)
        parser.add_argument("--server_learning_rate", type=float, default=None)
        parser.add_argument("--server_learning_rate_2", type=float, default=None)
        parser.add_argument("--target_round", type=int, default=None)
        parser.add_argument("--fixed", type=int, default=None, choices=[0, 1])

    def __init__(self, configs: Any, times: Any) -> None:
        super().__init__(configs=configs, times=times)
        for cid in range(self.num_clients):
            self.clients_personal_model_params[cid]["y_i"] = OrderedDict(
                (name, torch.zeros_like(value))
                for name, value in self.public_model_params.items()
            )

    def package(self, client_id: int) -> Dict[str, Any]:
        package = super().package(client_id=client_id)
        # y_i is emulated private client state; the official implementation
        # reinitializes w_i from the downloaded theta on every selected round.
        package["optimizer_state"] = {}
        if self.scheduler_mode != "iteration":
            package["scheduler_state"] = {}
        package["__wire__"] = ("regular_model_params",)
        return package

    def aggregate_client_updates(self, packages: "OrderedDict[int, dict]") -> None:
        learning_rate = (
            self.server_learning_rate_2
            if self.current_iter >= self.target_round
            else self.server_learning_rate
        )
        new_theta = OrderedDict()
        for name, theta in self.public_model_params.items():
            mean_delta = torch.stack(
                [package["delta"][name] for package in packages.values()]
            ).mean(dim=0)
            new_theta[name] = theta + learning_rate * mean_delta
        self._commit_global(new_params=new_theta)


class FedADMM_Client(FedADMMShared, tFL_Client):
    """Stateless worker emulating the paper's persistent FedADMM client."""

    def set_parameters(self, package: Dict[str, Any]) -> None:
        private_state = package["personal_model_params"]
        worker_package = dict(package)
        worker_package["personal_model_params"] = {}
        super().set_parameters(package=worker_package)

        self._previous_w = OrderedDict(
            (name, value.detach().cpu().clone())
            for name, value in self.model.named_parameters()
        )
        self._y = copy.deepcopy(private_state["y_i"])
        self._theta = copy.deepcopy(self._previous_w)

    def fit(self) -> None:
        self._set_worker_seed(seed=self._loader_seed(dataset_type="train"))
        loader = self.load_train_data()
        self.initialize_scheduler(steps_per_epoch=len(loader))
        self.model.to(self.device)
        self._move_optimizer_state_to_param_devices(optimizer=self.optimizer)
        self.model.train()

        local_epochs = (
            self.epochs
            if self.fixed
            else int(torch.randint(1, self.epochs + 1, ()).item())
        )
        for _ in range(local_epochs):
            for batch_x, batch_y, x_mark, y_mark in loader:
                self.optimizer.zero_grad(set_to_none=True)
                batch_x = batch_x.to(self.device, dtype=torch.float32)
                batch_y = batch_y.to(self.device, dtype=torch.float32)
                x_mark = x_mark.to(self.device, dtype=torch.float32)
                y_mark = y_mark.to(self.device, dtype=torch.float32)

                outputs = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
                loss = self.loss(outputs, batch_y)
                for name, parameter in self.model.named_parameters():
                    difference = parameter - self._theta[name].to(self.device)
                    loss = loss + torch.sum(
                        self._y[name].to(self.device) * difference
                        + self.rho * difference.square() / 2
                    )

                loss.backward()
                self.optimizer.step()
                self.step_scheduler_batch(
                    scheduler=self.scheduler,
                    batch_data=batch_x,
                )
            self.step_scheduler_epoch(scheduler=self.scheduler)

        if self.efficiency != "high":
            self.model.to("cpu")

    def package(self) -> Dict[str, Any]:
        package = super().package()
        current_w = package["regular_model_params"]
        current_y = OrderedDict(
            (
                name,
                self._y[name] + self.rho * (current_w[name] - self._theta[name]),
            )
            for name in current_w
        )
        package["delta"] = self.augmented_delta(
            previous_w=self._previous_w,
            previous_y=self._y,
            current_w=current_w,
            current_y=current_y,
            rho=self.rho,
        )
        package["personal_model_params"] = {
            "y_i": current_y,
        }
        package["regular_model_params"] = OrderedDict()
        package["__wire__"] = ("delta",)
        return package
