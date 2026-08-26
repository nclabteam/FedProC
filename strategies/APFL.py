import copy
from collections import OrderedDict
from typing import Any, Dict

import numpy as np
import torch
from torch.func import functional_call

from .pFL import pFL, pFL_Client


class APFLShared:
    @staticmethod
    def mix_parameters(
        personal_params: Any, global_params: Any, alpha: Any
    ) -> OrderedDict:
        return OrderedDict(
            (
                name,
                alpha * personal_param + (1.0 - alpha) * global_params[name].detach(),
            )
            for name, personal_param in personal_params.items()
        )

    @staticmethod
    def validate_alpha(alpha: float) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")


class APFL(APFLShared, pFL):
    """APFL: Adaptive Personalized Federated Learning (Deng et al., 2020)."""

    optional = {
        "alpha": 0.5,
    }

    @classmethod
    def args_update(cls, parser: Any) -> None:
        parser.add_argument("--alpha", type=float, default=None)

    def __init__(self, configs: Any, times: Any) -> None:
        super().__init__(configs=configs, times=times)
        self.validate_alpha(alpha=self.alpha)
        init_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        for cid in range(self.num_clients):
            self.clients_personal_model_params[cid]["model_per"] = {
                k: v.clone() for k, v in init_state.items()
            }
            self.clients_personal_model_params[cid]["alpha"] = self.alpha

    def aggregate_client_updates(self, packages: Any) -> None:
        self._commit_global(
            new_params=self.mean_models(
                models=[
                    package["regular_model_params"] for package in packages.values()
                ]
            )
        )


class APFL_Client(APFLShared, pFL_Client):
    """Client for APFL."""

    def __init__(self, configs: Any, times: Any, device: Any) -> None:
        super().__init__(configs=configs, times=times, device=device)
        self.model_per = copy.deepcopy(self.model)

    def set_parameters(self, package: Dict[str, Any]) -> None:
        super().set_parameters(package=package)
        self.model_per.load_state_dict(package["personal_model_params"]["model_per"])
        self.alpha = package["personal_model_params"]["alpha"]
        self.validate_alpha(alpha=self.alpha)

    def fit(self) -> None:
        self._set_worker_seed(seed=self._loader_seed(dataset_type="train"))
        loader = self.load_train_data()
        self.initialize_scheduler(steps_per_epoch=len(loader))
        optim_p = torch.optim.SGD(self.model_per.parameters(), lr=self.learning_rate)

        self.model.to(self.device)
        self._move_optimizer_state_to_param_devices(optimizer=self.optimizer)
        self.model_per.to(self.device)
        self.model.train()
        self.model_per.train()

        alpha_updated = False
        for _ in range(self.epochs):
            personal_lr = self.optimizer.param_groups[0]["lr"]
            optim_p.param_groups[0]["lr"] = personal_lr
            for batch_x, batch_y, x_mark, y_mark in loader:
                batch_x = batch_x.to(device=self.device, dtype=torch.float32)
                batch_y = batch_y.to(device=self.device, dtype=torch.float32)
                x_mark = x_mark.to(device=self.device, dtype=torch.float32)
                y_mark = y_mark.to(device=self.device, dtype=torch.float32)

                self.optimizer.zero_grad()
                optim_p.zero_grad()
                alpha = torch.tensor(
                    self.alpha,
                    device=self.device,
                    requires_grad=not alpha_updated,
                )
                mixed_params = self.mix_parameters(
                    personal_params=OrderedDict(self.model_per.named_parameters()),
                    global_params=OrderedDict(self.model.named_parameters()),
                    alpha=alpha,
                )
                out_p = functional_call(
                    self.model_per,
                    mixed_params,
                    (batch_x,),
                    {"x_mark": x_mark, "y_mark": y_mark},
                )
                loss_p = self.loss(out_p, batch_y)
                loss_p.backward()

                if not alpha_updated:
                    self.alpha = float(
                        np.clip(
                            self.alpha - personal_lr * alpha.grad.item(),
                            0.0,
                            1.0,
                        )
                    )
                    alpha_updated = True

                out_g = self.model(batch_x, x_mark=x_mark, y_mark=y_mark)
                loss_g = self.loss(out_g, batch_y)
                loss_g.backward()

                optim_p.step()
                self.optimizer.step()
                self.step_scheduler_batch(
                    scheduler=self.scheduler,
                    batch_data=batch_x,
                )

            self.step_scheduler_epoch(scheduler=self.scheduler)

        self.model.to("cpu")
        self.model_per.to("cpu")

    def package(self) -> Dict[str, Any]:
        result = super().package()
        result["personal_model_params"]["model_per"] = {
            k: v.detach().cpu().clone() for k, v in self.model_per.state_dict().items()
        }
        result["personal_model_params"]["alpha"] = self.alpha
        return result

    def evaluate_personalized(
        self,
        client_id: Any,
        global_params: Any,
        personal_params: Any,
        dataset_type: Any,
        current_iter: Any,
    ) -> float:
        self.id = client_id
        self.current_iter = current_iter
        self._load_private(client_id=client_id)
        self.model.load_state_dict(personal_params["model_per"], strict=False)
        self.model.load_state_dict(
            self.mix_parameters(
                personal_params=OrderedDict(
                    (name, personal_params["model_per"][name]) for name in global_params
                ),
                global_params=global_params,
                alpha=personal_params["alpha"],
            ),
            strict=False,
        )
        loader = (
            self.load_test_data() if dataset_type == "test" else self.load_train_data()
        )
        losses = self.calculate_loss(
            model=self.model,
            dataloader=loader,
            criterion=self.loss,
            device=self.device,
            offload_after=self.efficiency != "high",
        )
        return float(np.mean(losses))
