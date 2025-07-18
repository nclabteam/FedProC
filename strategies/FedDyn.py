import copy

import torch

from .base import Client, Server

optional = {
    "alpha": 0.1,
}


def args_update(parser):
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
        pass

    def aggregate_models(self):
        # Calculate the average model delta
        model_delta = self.reset_model(self.model)
        for client in self.client_data:
            client_model = client["model"]
            for server_param, client_param, delta_param in zip(
                self.model.parameters(),
                client_model.parameters(),
                model_delta.parameters(),
            ):
                delta_param.data += (client_param - server_param) / self.num_clients

        # Update the server state
        for state_param, delta_param in zip(
            self.server_state.parameters(), model_delta.parameters()
        ):
            state_param.data -= self.alpha * delta_param

        # Update the server model
        self.model = self.reset_model(self.model)
        for client in self.client_data:
            for server_param, client_param in zip(
                self.model.parameters(), client["model"].parameters()
            ):
                server_param.data += client_param.data.clone() / self.num_join_clients

        # Apply the server state to the model
        for server_param, state_param in zip(
            self.model.parameters(), self.server_state.parameters()
        ):
            server_param.data -= (1 / self.alpha) * state_param


class FedDyn_Client(Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_model_vector = None
        self.old_grad = torch.zeros_like(
            self.model_parameter_vector(copy.deepcopy(self.model))
        )

    def train_one_epoch(
        self, model, dataloader, optimizer, criterion, scheduler, device
    ):
        model.to(device)
        model.train()
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            if self.global_model_vector is not None:
                self.global_model_vector = self.global_model_vector.to(device)
                self.old_grad = self.old_grad.to(device)
                v1 = self.model_parameter_vector(model)
                loss += self.alpha / 2 * torch.norm(v1 - self.global_model_vector, 2)
                loss -= torch.dot(v1, self.old_grad)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if self.global_model_vector is not None:
            v1 = self.model_parameter_vector(model).detach()
            self.old_grad = self.old_grad - self.alpha * (v1 - self.global_model_vector)
        scheduler.step()

        self.model.to("cpu")
        self.global_model_vector = None
        self.old_grad = self.old_grad.to("cpu")

    def receive_from_server(self, data):
        super().receive_from_server(data)
        self.global_model_vector = (
            self.model_parameter_vector(data["model"]).detach().clone()
        )

    @staticmethod
    def model_parameter_vector(model):
        return torch.cat([p.view(-1) for p in model.parameters()], dim=0)
