import copy
from typing import Any, Dict

import torch
import torch.nn.functional as F

from .tFL import tFL, tFL_Client


class MOON(tFL):
    """Model-Contrastive Federated Learning (Li et al., CVPR 2021)."""

    optional = {
        "mu": 1.0,
        "temperature": 0.5,
    }

    @classmethod
    def args_update(cls, parser: Any) -> None:
        parser.add_argument("--mu", type=float, default=None)
        parser.add_argument("--temperature", type=float, default=None)

    def __init__(self, configs: Any, times: Any) -> None:
        super().__init__(configs=configs, times=times)
        init_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        for cid in range(self.num_clients):
            self.clients_personal_model_params[cid]["prev_model_state"] = {
                k: v.clone() for k, v in init_state.items()
            }


class MOON_Client(tFL_Client):
    def set_parameters(self, package: Dict[str, Any]) -> None:
        self._global_model_params = copy.deepcopy(package["regular_model_params"])
        self._prev_model_params = package["personal_model_params"]["prev_model_state"]
        super().set_parameters(package=package)

    def package(self) -> Dict[str, Any]:
        out = super().package()
        # Persist current post-training model as prev_model for next round (w_i^t → w_i^{t-1})
        out["personal_model_params"]["prev_model_state"] = {
            k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
        }
        return out

    def train_one_epoch(
        self,
        model: Any,
        dataloader: Any,
        optimizer: Any,
        criterion: Any,
        scheduler: Any,
        device: Any,
        offload_after: Any = True,
    ) -> None:
        model.to(device)
        self._move_optimizer_state_to_param_devices(optimizer=optimizer)

        # Frozen reference models for contrastive loss
        global_model = copy.deepcopy(model)
        global_model.load_state_dict(self._global_model_params, strict=False)
        global_model.to(device).eval()

        prev_model = copy.deepcopy(model)
        prev_model.load_state_dict(self._prev_model_params, strict=False)
        prev_model.to(device).eval()

        model.train()
        for batch_x, batch_y, x_mark, y_mark in dataloader:
            optimizer.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            x_mark = x_mark.to(device)
            y_mark = y_mark.to(device)

            # Supervised loss ℓ_sup (Eq. 4)
            outputs = model(batch_x, x_mark=x_mark, y_mark=y_mark)
            loss_sup = criterion(outputs, batch_y)

            # Representations z, z_glob, z_prev (flattened output as proxy for R_w(x))
            z = outputs.flatten(start_dim=1)
            with torch.no_grad():
                z_glob = global_model(batch_x, x_mark=x_mark, y_mark=y_mark).flatten(
                    start_dim=1
                )
                z_prev = prev_model(batch_x, x_mark=x_mark, y_mark=y_mark).flatten(
                    start_dim=1
                )

            # Model-contrastive loss ℓ_con (Eq. 3)
            sim_glob = F.cosine_similarity(z, z_glob, dim=1) / self.temperature
            sim_prev = F.cosine_similarity(z, z_prev, dim=1) / self.temperature
            loss_con = -torch.log(
                torch.exp(sim_glob) / (torch.exp(sim_glob) + torch.exp(sim_prev))
            ).mean()

            # Total loss ℓ = ℓ_sup + μ * ℓ_con (Eq. 4)
            loss = loss_sup + self.mu * loss_con
            loss.backward()
            optimizer.step()
            self.step_scheduler_batch(
                scheduler=scheduler,
                batch_data=batch_x,
            )

        self.step_scheduler_epoch(scheduler=scheduler)
        if offload_after:
            model.to("cpu")
        del global_model, prev_model
