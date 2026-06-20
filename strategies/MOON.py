import copy
from typing import Any, Dict

import torch
import torch.nn.functional as F

from .tFL import tFL, tFL_Client


class MOON(tFL):
    optional = {
        "mu": 1.0,
        "temperature": 0.5,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--mu", type=float, default=None)
        parser.add_argument("--temperature", type=float, default=None)


class MOON_Client(tFL_Client):
    def set_parameters(self, package: Dict[str, Any]) -> None:
        self.id = package["client_id"]
        self.current_iter = package["current_iter"]
        self._load_private(self.id)
        # Load global model as the starting point for local training (w^t)
        self.model.load_state_dict(package["regular_model_params"], strict=False)
        if package["optimizer_state"]:
            self.optimizer.load_state_dict(package["optimizer_state"])
        else:
            self.optimizer.load_state_dict(self.init_optimizer_state)
        if package["scheduler_state"]:
            self.scheduler.load_state_dict(package["scheduler_state"])
        else:
            self.scheduler.load_state_dict(self.init_scheduler_state)
        # [§Method] — save w^t (global model) as positive contrastive reference
        self._global_model_params = copy.deepcopy(package["regular_model_params"])
        # [§Method] — restore w_i^{t-1} (previous local model); on first round fall back to global
        personal = package["personal_model_params"]
        self._prev_model_params = personal.get(
            "prev_model_state",
            copy.deepcopy(package["regular_model_params"]),
        )

    def package(self, train_time: float) -> Dict[str, Any]:
        out = super().package(train_time)
        # Persist current post-training model as prev_model for next round (w_i^t → w_i^{t-1})
        out["personal_model_params"]["prev_model_state"] = {
            k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
        }
        return out

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

        # [§Method] — frozen reference models for contrastive loss
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

            # [eq.3] — supervised loss ℓ_sup
            outputs = model(batch_x, x_mark=x_mark, y_mark=y_mark)
            loss_sup = criterion(outputs, batch_y)

            # [§Method] — representations z, z_glob, z_prev
            # TSF adaptation: use flattened output as representation (no projection head)
            z = outputs.flatten(start_dim=1)
            with torch.no_grad():
                z_glob = global_model(batch_x, x_mark=x_mark, y_mark=y_mark).flatten(
                    start_dim=1
                )
                z_prev = prev_model(batch_x, x_mark=x_mark, y_mark=y_mark).flatten(
                    start_dim=1
                )

            # [eq.1] — model-contrastive loss ℓ_con (NT-Xent style)
            sim_glob = F.cosine_similarity(z, z_glob, dim=1) / self.temperature
            sim_prev = F.cosine_similarity(z, z_prev, dim=1) / self.temperature
            loss_con = -torch.log(
                torch.exp(sim_glob) / (torch.exp(sim_glob) + torch.exp(sim_prev))
            ).mean()

            # [eq.2] — total loss ℓ = ℓ_sup + μ * ℓ_con
            loss = loss_sup + self.mu * loss_con
            loss.backward()
            optimizer.step()

        scheduler.step()
        if offload_after:
            model.to("cpu")
        del global_model, prev_model
