import gc

import torch.nn as nn

from .tFL import tFL, tFL_Client


class nFL(tFL):
    """
    No-Federated-Learning base.

    For strategies that do not perform federated model aggregation
    (e.g. Centralized, LocalOnly).  Server-model-dependent methods are
    no-ops so subclasses only need to override what they actually use.
    """

    compulsory = {"exclude_server_model_processes": True}

    def initialize_model(self, *args, **kwargs):
        pass

    def send_to_clients(self, *args, **kwargs):
        pass

    def receive_from_clients(self, *args, **kwargs):
        pass

    def aggregate_models(self, *args, **kwargs):
        pass

    def calculate_aggregation_weights(self, *args, **kwargs):
        pass

    def get_model_info(self) -> None:
        for client in self.clients:
            if hasattr(client, "model") and isinstance(client.model, nn.Module):
                dl = client.load_train_data()
                client.summarize_model(dataloader=dl)
                del dl
                gc.collect()


class nFL_Client(tFL_Client):
    """Passthrough — same as tFL_Client."""
