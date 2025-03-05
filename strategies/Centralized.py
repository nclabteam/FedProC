from collections import deque

import ray

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
        if self.parallel:  # Use parallel execution with Ray
            futures = []
            idle_workers = deque(range(self.num_workers))
            job_map = {}

            i = 0
            while i < len(self.client_data) or len(futures) > 0:
                while i < len(self.client_data) and len(idle_workers) > 0:
                    worker_id = idle_workers.popleft()
                    client = self.client_data[i]
                    train_loader = client["dataloader"]

                    # Parallelized `train_one_epoch` execution
                    future = train_one_epoch_remote.remote(
                        model=self.model,
                        dataloader=train_loader,
                        optimizer=self.optimizer,
                        criterion=self.loss,
                        scheduler=self.scheduler,
                        device=self.device,
                        epochs=self.epochs,
                    )

                    job_map[future] = (client, worker_id)
                    futures.append(future)
                    i += 1

                if len(futures) > 0:
                    all_finished, futures = ray.wait(futures)
                    for finished in all_finished:
                        client, worker_id = job_map[finished]
                        model_state, optimizer_state = ray.get(finished)

                        # Assign updated model & optimizer back
                        self.model.load_state_dict(model_state)
                        self.optimizer.load_state_dict(optimizer_state)

                        idle_workers.append(worker_id)

        else:  # Run in serial mode
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

    def evaluate_personalization_loss(self, *args, **kwargs):
        pass

    def aggregate_models(self):
        pass

    def calculate_aggregation_weights(self):
        pass


class Centralized_Client(Client):
    def variables_to_be_sent(self):
        return {"dataloader": self.load_train_data()}


@ray.remote
def train_one_epoch_remote(
    model, dataloader, optimizer, criterion, scheduler, device, epochs
):
    model.train()
    for _ in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            inputs, targets = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model.state_dict(), optimizer.state_dict()
