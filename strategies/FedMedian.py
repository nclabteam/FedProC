import torch

from .base import Client, Server


class FedMedian(Server):

    def calculate_aggregation_weights(self):
        pass

    def aggregate_models(self):
        self.model = self.reset_model(self.model)
        for name, param in self.model.named_parameters():
            layers = torch.stack(
                [client["model"].state_dict()[name] for client in self.client_data]
            )
            param.data = torch.median(layers, dim=0).values.clone()


class FedMedian_Client(Client):
    def variables_to_be_sent(self):
        return {"model": self.model}
