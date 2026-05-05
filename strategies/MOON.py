import copy

import torch
import torch.nn.functional as F

from .base import Client, Server


class MOON(Server):
    optional = {
        "mu": 1.0,
        "temperature": 0.5,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--mu", type=float, default=None)
        parser.add_argument("--temperature", type=float, default=None)


class MOON_Client(Client):
    def receive_from_server(self, data):
        # [§Method] — save w_i^{t-1} (previous local model) before overwriting
        self.prev_model = copy.deepcopy(self.model).to("cpu")
        # [§Method] — save w^t (global model) as positive reference
        self.global_model = copy.deepcopy(data["model"]).to("cpu")
        self.update_model_params(old=self.model, new=data["model"])

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
        global_model = copy.deepcopy(self.global_model).to(device)
        prev_model = copy.deepcopy(self.prev_model).to(device)
        global_model.eval()
        prev_model.eval()

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
