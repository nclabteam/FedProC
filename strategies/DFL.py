import logging
import time
from concurrent.futures import ThreadPoolExecutor

import ray
import torch

from topologies import TOPOLOGIES

from .base import Client, Server

optional = {
    "topology": "FullyConnected",
}

compulsory = {
    "save_local_model": True,
    "exclude_server_model_processes": True,
}


def args_update(parser):
    parser.add_argument("--topology", type=str, default=None, choices=TOPOLOGIES)


class DFL(Server):
    def __init__(self, configs, times):
        self.set_configs(configs=configs, times=times)
        self.mkdir()

        self.num_gpus = len(self.device_id.split(","))
        configs.parallel = True if self.num_gpus > 1 else False
        self.parallel = configs.parallel
        ray.init(
            num_gpus=self.num_gpus,
            ignore_reinit_error=True,
            logging_level=logging.ERROR,
            log_to_driver=False,
        )

        self.metrics = {
            "time_per_iter": [],
            "personal_avg_train_loss": [],
            "personal_avg_test_loss": [],
        }
        self.name = "  ORCHES  "
        self.make_logger(name=self.name, path=self.log_path)
        self.get_topology()
        self.initialize_clients()

    def get_topology(self):
        self.topology = getattr(__import__("topologies"), self.topology)(
            num_nodes=self.num_clients
        ).neighbors
        self.configs.__dict__["neighbors"] = self.topology

    def initialize_model(self, *args, **kwargs):
        pass

    def initialize_optimizer(self, *args, **kwargs):
        pass

    def initialize_scheduler(self, *args, **kwargs):
        pass

    def initialize_loss(self, *args, **kwargs):
        pass

    def select_clients(self, *args, **kwargs):
        self.selected_clients = self.clients

    def receive_from_clients(self):
        for node in self.clients:
            start = time.time()
            receive_mb = 0
            all_to_be_received = {}
            for key, value in node.variables_to_be_sent().items():
                all_to_be_received[key] = [value]

            for neighbor in node.neighbors:
                neighbor = self.clients[neighbor]
                to_be_received = neighbor.variables_to_be_sent()
                for key, value in to_be_received.items():
                    if isinstance(value, list):
                        value = value[node.id]
                    receive_mb += self.get_size(value)
                    all_to_be_received[key].append(value)
            node.metrics["receive_mb"].append(receive_mb)
            node.metrics["send_time"].append(time.time() - start)
            node.receive_from_server(all_to_be_received)

    def send_to_clients(self, *args, **kwargs):
        pass

    def variables_to_be_sent(self, *args, **kwargs):
        pass

    def calculate_aggregation_weights(self, *args, **kwargs):
        if self.parallel:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [
                    executor.submit(node.calculate_aggregation_weights)
                    for node in self.clients
                ]
                for future in futures:
                    future.result()
        else:
            for node in self.clients:
                node.calculate_aggregation_weights()

    def aggregate_models(self, *args, **kwargs):
        if self.parallel:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [
                    executor.submit(node.aggregate_models) for node in self.clients
                ]
                for future in futures:
                    future.result()
        else:
            for node in self.clients:
                node.aggregate_models()


class DFL_Client(Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.neighbors = kwargs["configs"].neighbors[self.id]
        self.logger.info(f"Neighbors: {self.neighbors}")
        self.metrics["receive_mb"] = []

    def receive_from_server(self, data):
        self.scores = data["score"]
        self.models = data["model"]

    def calculate_aggregation_weights(self):
        self.weights = torch.tensor(self.scores).to(self.device) / sum(self.scores)

    def aggregate_models(self):
        model = self.reset_model(self.model)
        for client, weight in zip(self.models, self.weights):
            for global_param, local_param in zip(
                model.parameters(), client.parameters()
            ):
                global_param.data.add_(local_param.data, alpha=weight)
        self.update_model_params(old=self.model, new=model)
