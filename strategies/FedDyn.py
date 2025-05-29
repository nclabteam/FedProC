import copy

import torch

from .base import Client, Server

optional = {
    "alpha": 0.1,
}


def args_update_feddyn(parser):
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Alpha hyperparameter for FedDyn (regularization strength)",
    )


class FedDyn(Server):
    """
    Paper: https://arxiv.org/abs/2111.04263
    Source: https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/servers/serverdyn.py
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_state = self.reset_model(self.model)

    def calculate_aggregation_weights(self):
        model_delta = self.reset_model(self.model)

        for client in self.client_data:
            client_model = client["model"]
            for server_param, client_param, delta_param in zip(
                self.model.parameters(),
                client_model.parameters(),
                model_delta.parameters(),
            ):
                delta_param.data += (client_param - server_param) / self.num_clients

        for state_param, delta_param in zip(
            self.server_state.parameters(), model_delta.parameters()
        ):
            state_param.data -= self.alpha * delta_param

    def aggregate_models(self):
        self.model = self.reset_model(self.model)

        for client in self.client_data:
            for server_param, client_param in zip(
                self.model.parameters(), client["model"].parameters()
            ):
                server_param.data += client_param.data.clone() / self.num_join_clients

        for server_param, state_param in zip(
            self.model.parameters(), self.server_state.parameters()
        ):
            server_param.data -= (1 / self.alpha) * state_param


class FedDyn_Client(Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_model_vector = None
        old_grad = copy.deepcopy(self.model)
        old_grad = model_parameter_vector(old_grad)
        self.old_grad = torch.zeros_like(old_grad)

    def train_one_epoch(
        self, model, dataloader, optimizer, criterion, scheduler, device
    ):
        model.train()
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            if self.global_model_vector is not None:
                v1 = model_parameter_vector(model)
                loss += self.alpha / 2 * torch.norm(v1 - self.global_model_vector, 2)
                loss -= torch.dot(v1, self.old_grad)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if self.global_model_vector is not None:
            v1 = model_parameter_vector(model).detach()
            self.old_grad = self.old_grad - self.alpha * (v1 - self.global_model_vector)
        scheduler.step()

    def receive_from_server(self, data):
        super().receive_from_server(data)
        self.global_model_vector = (
            model_parameter_vector(data["model"]).detach().clone()
        )


def model_parameter_vector(model):
    param = [p.view(-1) for p in model.parameters()]
    return torch.cat(param, dim=0)
