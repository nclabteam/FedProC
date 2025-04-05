from .base import Client, Server

compulsory = {
    "save_local_model": True,
    "exclude_server_model_processes": True,
}


class LocalOnly(Server):
    def receive_from_clients(self, *args, **kwargs):
        pass

    def calculate_aggregation_weights(self, *args, **kwargs):
        pass

    def aggregate_models(self, *args, **kwargs):
        pass

    def send_to_clients(self, *args, **kwargs):
        pass

    def evaluate_generalization_loss(self, *args, **kwargs):
        pass

    def get_model_info(self):
        for client in self.clients:
            client.summarize_model(dataloader=client.load_train_data())


class LocalOnly_Client(Client):
    pass
