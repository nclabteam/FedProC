import os
import sys
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.base import Client, Server
from strategies.FedAvg import FedAvg, FedAvg_Client
from strategies.LocalOnly import LocalOnly, LocalOnly_Client


class DummyClient:
    def __init__(self, client_id):
        self.id = client_id
        self.input_channels = 2
        self.output_channels = 2
        self.received = []

    def receive_from_server(self, package):
        self.received.append(package)

    def send_to_server(self):
        return {"client_id": self.id}


class TestStrategies(unittest.TestCase):
    def make_configs(self, save_path):
        return Namespace(
            save_path=save_path,
            num_clients=3,
            join_ratio=1.0,
            random_join_ratio=False,
            device="cpu",
            device_id="",
            num_workers=0,
            save_local_model=False,
            exclude_server_model_processes=True,
            skip_eval_train=False,
            eval_gap=1,
            patience=0,
        )

    def build_server(self, strategy_cls):
        with tempfile.TemporaryDirectory() as tmpdir:
            configs = self.make_configs(tmpdir)

            def fake_client_init(self, configs, id, times):
                self.id = id
                self.input_channels = 2
                self.output_channels = 2
                self.train_samples = 1

            with (
                patch("strategies.base.ray.init"),
                patch.object(
                    Server,
                    "initialize_model",
                    lambda self: setattr(self, "model", "model"),
                ),
                patch.object(Server, "get_model_info", lambda self: None),
                patch.object(Server, "make_logger", lambda self, name, path: None),
                patch.object(Client, "__init__", fake_client_init),
            ):
                server = strategy_cls(configs=configs, times=0)
        return server

    def test_localonly_uses_strategy_specific_client_class(self):
        server = self.build_server(LocalOnly)
        self.assertTrue(
            all(isinstance(client, LocalOnly_Client) for client in server.clients)
        )
        self.assertFalse(server.parallel)
        self.assertEqual(server.num_gpus, 0)

    def test_fedavg_uses_strategy_specific_client_class(self):
        server = self.build_server(FedAvg)
        self.assertTrue(
            all(isinstance(client, FedAvg_Client) for client in server.clients)
        )
        self.assertFalse(server.parallel)
        self.assertEqual(server.num_gpus, 0)

    def test_fedavg_one_round_smoke(self):
        server = object.__new__(FedAvg)
        server.random_join_ratio = False
        server.num_join_clients = 2
        server.current_num_join_clients = 2
        server.num_clients = 3
        server.clients = [DummyClient(0), DummyClient(1), DummyClient(2)]
        server.model = "model"
        server.metrics = {"send_mb": []}
        server.logger = None
        server.selected_clients = []
        server.get_size = lambda _: 1.5

        Server.select_clients(server)
        self.assertEqual(len(server.selected_clients), 2)

        Server.send_to_clients(server)
        self.assertEqual(server.metrics["send_mb"], [4.5])
        self.assertTrue(all(client.received for client in server.clients))

        Server.receive_from_clients(server)
        self.assertEqual(len(server.client_data), 2)
        self.assertEqual(
            sorted(item["client_id"] for item in server.client_data),
            sorted(client.id for client in server.selected_clients),
        )

    def test_localonly_round_hooks_are_noops(self):
        local_only = object.__new__(LocalOnly)
        self.assertIsNone(LocalOnly.receive_from_clients(local_only))
        self.assertIsNone(LocalOnly.calculate_aggregation_weights(local_only))
        self.assertIsNone(LocalOnly.aggregate_models(local_only))
        self.assertIsNone(LocalOnly.send_to_clients(local_only))
        self.assertIsNone(LocalOnly.evaluate_generalization_loss(local_only))
        self.assertIsNone(LocalOnly.initialize_model(local_only))


if __name__ == "__main__":
    unittest.main()
