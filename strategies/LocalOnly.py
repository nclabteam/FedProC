from .base import Client, Server

compulsory = {
    "save_local_model": True,
}


class LocalOnly(Server):
    def initialize_model(self):
        pass

    def receive_from_clients(self):
        pass

    def calculate_aggregation_weights(self):
        pass

    def aggregate_models(self):
        pass

    def send_to_clients(self):
        pass

    def evaluate_generalization_loss(self, *args, **kwargs):
        pass

    def get_model_info(self):
        for client in self.clients:
            client.summarize_model(dataloader=client.load_train_data())


class LocalOnly_Client(Client):
    pass
