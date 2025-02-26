from layers import RevIN

from .base import Client, Server

compulsory = {
    "save_local_model": True,
}


class FedRevIN(Server):
    """
    FedBN of time series. Exclude RevIN layer in aggregation
    """

    pass


class FedRevIN_Client(Client):
    def update_model_params(self, model):
        for new_param, (name, old_param) in zip(
            model.parameters(), self.model.named_parameters()
        ):
            # Check if the layer is an instance of RevIN
            layer = getattr(self.model, name.split(".")[0], None)
            if isinstance(layer, RevIN):
                continue
            old_param.data = new_param.data.clone()
