import sys

from .base import Client, Server


class Centralized(Server):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.get_loss()
        self.get_optimizer()
        self.get_scheduler()

    def train_clients(self):
        self.model.train()
        for client in self.clients:
            train_loader = client.load_train_data()
            client.metrics["send_mb"].append(self.get_size(train_loader))
            for _ in range(self.epochs):
                self.train_one_epoch(
                    model=self.model,
                    dataloader=train_loader,
                    optimizer=self.optimizer,
                    criterion=self.loss,
                    scheduler=self.scheduler,
                    device=self.device,
                )

    def send_to_clients(self):
        pass

    def receive_from_clients(self):
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
