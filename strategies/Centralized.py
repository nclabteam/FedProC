from .base import Client, Server


class Centralized(Server):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialize_loss()
        self.initialize_optimizer()
        self.initialize_scheduler()

        self.receive_from_clients()
        self.fix_results()

    def train_clients(self):
        self.model.train()
        for client in self.client_data:
            train_loader = client["dataloader"]
            for _ in range(self.epochs):
                self.train_one_epoch(
                    model=self.model,
                    dataloader=train_loader,
                    optimizer=self.optimizer,
                    criterion=self.loss,
                    scheduler=self.scheduler,
                    device=self.device,
                )

    def receive_from_clients(self):
        self.client_data = []
        for client in self.clients:
            self.client_data.append(client.send_to_server())

    def send_to_clients(self):
        pass

    def evaluate_personalization_valset(self):
        pass

    def evaluate_personalization_trainset(self):
        pass

    def evaluate_personalization_testset(self):
        pass

    def aggregate_models(self):
        pass

    def calculate_aggregation_weights(self):
        pass


class Centralized_Client(Client):
    def variables_to_be_sent(self):
        return {"dataloader": self.load_train_data()}
