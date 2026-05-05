import copy

# import numpy as np
import torch
import torch.nn.functional as F

from losses import KLDivergence

from .base import Client, Server


class FML(Server):

    optional = {
        "alpha": 0.9,
        "beta": 0.1,
    }

    compulsory = {
        "save_local_model": True,
    }

    @classmethod
    def args_update(cls, parser):
        parser.add_argument("--alpha", type=float, default=None)
        parser.add_argument("--beta", type=float, default=None)

    def calculate_aggregation_weights(self):
        self.weights = torch.tensor([1 / len(self.client_data)] * len(self.client_data))


class FML_Client(Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.metrics["train_loss_g"] = []
        # self.metrics["test_loss_g"] = []

        self.model_g = copy.deepcopy(self.model)
        obj = self._get_objective_function("optimizers", "Adam")
        self.optimizer_g = obj(params=self.model_g.parameters(), configs=self.configs)
        self.KL = KLDivergence()

    def variables_to_be_sent(self):
        return {"model": self.model_g}

    def train_one_epoch(self, dataloader, *args, offload_after=True, **kwargs):
        self.model.to(self.device)
        self.model_g.to(self.device)
        self._move_optimizer_state_to_param_devices(self.optimizer)
        self._move_optimizer_state_to_param_devices(self.optimizer_g)
        self.model.train()
        self.model_g.train()
        for batch in dataloader:
            batch_x = batch[0].float().to(self.device)
            batch_y = batch[1].float().to(self.device)
            x_mark = batch[2].to(self.device) if len(batch) > 2 else None
            y_mark = batch[3].to(self.device) if len(batch) > 3 else None
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

    def receive_from_server(self, data):
        self.update_model_params(old=self.model_g, new=data["model"])

    # def get_train_loss(self):
    #     results = super().get_train_loss()
    #     losses = self.calculate_loss(
    #         model=self.model_g,
    #         dataloader=self.load_train_data(),
    #         criterion=self.loss,
    #         device=self.device,
    #     )
    #     losses = np.mean(losses)
    #     self.metrics["train_loss_g"].append(losses)
    #     return results

    # def get_test_loss(self):
    #     results = super().get_test_loss()
    #     losses = self.calculate_loss(
    #         model=self.model_g,
    #         dataloader=self.load_test_data(),
    #         criterion=self.loss,
    #         device=self.device,
    #     )
    #     losses = np.mean(losses)
    #     self.metrics["test_loss_g"].append(losses)
    #     return results
