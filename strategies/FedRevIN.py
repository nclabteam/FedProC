import torch.nn as nn

from layers import RevIN

from .base import Client, Server

compulsory = {
    "save_local_model": True,
}


class FedRevIN(Server):
    """
    FedBN
    """

    pass


class FedRevIN_Client(Client):
    def initialize_local(self, model):
        for new_param, (name, old_param) in zip(
            model.parameters(), self.model.named_parameters()
        ):
            if name == "rev":
                continue
            old_param.data = new_param.data.clone()
