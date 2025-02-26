from .base import Client, Server

compulsory = {
    "save_local_model": True,
}


class LocalOnly(Server):
    def receive_from_clients(self):
        pass

    def calculate_aggregation_weights(self):
        pass

    def aggregate_models(self):
        pass

    def send_to_clients(self):
        pass

    def evaluate_generalization_loss(self):
        pass


class LocalOnly_Client(Client):
    pass
