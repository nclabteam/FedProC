import torch.nn as nn

from layers import RevIN

from .base import Client, Server, SharedMethods

compulsory = {
    "save_local_model": True,
}


class ModelWithRevIN(nn.Module):
    def __init__(self, base_model, in_channels):
        super(ModelWithRevIN, self).__init__()
        self.rev = RevIN(in_channels)
        self.base_model = base_model

    def forward(self, x):
        x = self.rev(x, "norm")
        x = self.base_model(x)
        x = self.rev(x, "denorm")
        return x


class FedRevINShared(SharedMethods):
    def ensure_revin_wrapped(self):
        """
        Ensures the model is wrapped with RevIN if not already present.
        """
        # Check if the model already has a RevIN wrapper
        for module in self.model.modules():
            if isinstance(module, RevIN):
                return  # Return the model as is if RevIN exists

        # If no RevIN exists, wrap the model
        self.model = ModelWithRevIN(self.model, self.input_channels).to(self.device)

    def initialize_model(self, *args, **kwargs):
        super().initialize_model(*args, **kwargs)
        self.ensure_revin_wrapped()


class FedRevIN(Server, FedRevINShared):
    pass


class FedRevIN_Client(Client, FedRevINShared):
    def update_model_params(self, old, new):
        for new_param, (name, old_param) in zip(
            new.parameters(), old.named_parameters()
        ):
            # Check if the layer is an instance of RevIN
            layer = getattr(old, name.split(".")[0], None)
            if isinstance(layer, RevIN):
                continue
            old_param.data = new_param.data.clone()
