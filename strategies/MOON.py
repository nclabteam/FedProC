import copy
from typing import Any, Dict

import torch
import torch.nn.functional as F

from .tFL import tFL, tFL_Client


class MOON(tFL):
    """Model-Contrastive Federated Learning (Li et al., CVPR 2021).

    Each client's local loss = supervised loss + μ * model-contrastive loss.
    The contrastive term pulls the current local representation toward the
    global model's representation (positive) and away from the previous local
    model's representation (negative).

    Default hyperparameters from the paper: temperature τ = 0.5, μ ∈ {0.1,1,5,10}
    (dataset-dependent; 1.0 used as a reasonable default here).

    TSF adaptation: the paper uses a projection head R_w(x) before the output
    layer as the representation. For TSF models that lack a projection head,
    we use the flattened model output directly.

    Reference: arXiv:2103.16257.
    """

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
        # Save contrastive references before super() loads the global model params
        self._global_model_params = copy.deepcopy(package["regular_model_params"])
        personal = package["personal_model_params"]
        self._prev_model_params = personal.get(
            "prev_model_state",
            copy.deepcopy(package["regular_model_params"]),
        )
        super().set_parameters(package)

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
                z_glob = global_model(batch_x, x_mark=x_mark, y_mark=y_mark).flatten(start_dim=1)
                z_prev = prev_model(batch_x, x_mark=x_mark, y_mark=y_mark).flatten(start_dim=1)

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

        scheduler.step()
        if offload_after:
            model.to("cpu")
        del global_model, prev_model
