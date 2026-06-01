import os
import sys
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imports are deferred to test methods / build_server to survive module reloads
# from test_lazy_registries.py


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
            exclude_server_model_processes=True,
            skip_eval_train=False,
            eval_gap=1,
            patience=0,
        )

    def build_server(self, strategy_cls):
        # Reimport to get fresh class objects (test_lazy_registries may reload modules)
        from strategies.nFL import nFL as _nFL
        from strategies.pFL import pFL as _pFL
        from strategies.tFL import tFL as _tFL
        from strategies.tFL import tFL_Client as _tFL_Client

        with tempfile.TemporaryDirectory() as tmpdir:
            configs = self.make_configs(tmpdir)

            def fake_client_init(self, configs, id, times):
                self.id = id
                self.input_channels = 2
                self.output_channels = 2
                self.train_samples = 1

            with (
                patch("strategies.base.ray.init"),
                patch("strategies.tFL.ray.init"),
                patch.object(
                    _tFL,
                    "initialize_model",
                    lambda self: setattr(self, "model", "model"),
                ),
                patch.object(_tFL, "get_model_info", lambda self: None),
                patch.object(_pFL, "get_model_info", lambda self: None),
                patch.object(_nFL, "get_model_info", lambda self: None),
                patch.object(_tFL, "make_logger", lambda self, name, path: None),
                patch.object(_tFL_Client, "__init__", fake_client_init),
            ):
                server = strategy_cls(configs=configs, times=0)
        return server

    def test_localonly_uses_strategy_specific_client_class(self):
        from strategies.LocalOnly import LocalOnly
        from strategies.LocalOnly import LocalOnly_Client as CurrentClient

        server = self.build_server(LocalOnly)
        self.assertTrue(
            all(isinstance(client, CurrentClient) for client in server.clients)
        )
        self.assertFalse(server.parallel)
        self.assertEqual(server.num_gpus, 0)

    def test_fedavg_uses_strategy_specific_client_class(self):
        from strategies.FedAvg import FedAvg
        from strategies.FedAvg import FedAvg_Client as CurrentClient

        server = self.build_server(FedAvg)
        self.assertTrue(
            all(isinstance(client, CurrentClient) for client in server.clients)
        )
        self.assertFalse(server.parallel)
        self.assertEqual(server.num_gpus, 0)

    def test_fedavg_one_round_smoke(self):
        from strategies.FedAvg import FedAvg
        from strategies.tFL import tFL

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

        tFL.select_clients(server)
        self.assertEqual(len(server.selected_clients), 2)

        tFL.send_to_clients(server)
        self.assertEqual(server.metrics["send_mb"], [4.5])
        self.assertTrue(all(client.received for client in server.clients))

        tFL.receive_from_clients(server)
        self.assertEqual(len(server.client_data), 2)
        self.assertEqual(
            sorted(item["client_id"] for item in server.client_data),
            sorted(client.id for client in server.selected_clients),
        )

    def test_fedcross_sends_slot_model_to_each_selected_client(self):
        from strategies.FedCross import FedCross

        server = object.__new__(FedCross)
        clients = [DummyClient(0), DummyClient(1), DummyClient(2)]
        server.clients = clients
        server.selected_clients = [clients[2], clients[0]]
        server.w_locals = ["slot_0", "slot_1"]
        server.w_locals_num = 2
        server.current_iter = 7
        server.metrics = {"send_mb": []}
        server.get_size = lambda _: 1.25

        FedCross.send_to_clients(server)

        self.assertEqual(clients[2].received, [{"model": "slot_0"}])
        self.assertEqual(clients[0].received, [{"model": "slot_1"}])
        self.assertEqual(clients[1].received, [])
        self.assertEqual(clients[2].current_iter, 7)
        self.assertEqual(clients[0].current_iter, 7)
        self.assertEqual(server.metrics["send_mb"], [2.5])

    def test_localonly_round_hooks_are_noops(self):
        from strategies.LocalOnly import LocalOnly

        local_only = object.__new__(LocalOnly)
        self.assertIsNone(LocalOnly.receive_from_clients(local_only))
        self.assertIsNone(LocalOnly.calculate_aggregation_weights(local_only))
        self.assertIsNone(LocalOnly.aggregate_models(local_only))
        self.assertIsNone(LocalOnly.send_to_clients(local_only))
        self.assertIsNone(LocalOnly.initialize_model(local_only))

    def test_fedpaq_uses_strategy_specific_client_class(self):
        from strategies.FedPAQ import FedPAQ
        from strategies.FedPAQ import FedPAQ_Client as CurrentClient

        server = self.build_server(FedPAQ)
        self.assertTrue(
            all(isinstance(client, CurrentClient) for client in server.clients)
        )
        self.assertTrue(FedPAQ.compulsory.get("return_diff"))

    def test_fedpaq_quantize_tensor(self):
        import torch

        from strategies.FedPAQ import FedPAQ_Client

        # Test unbiased stochastic quantization on constant tensor
        tensor = torch.ones(100000)
        quantized = FedPAQ_Client.quantize_tensor(tensor, 4)
        mean_diff = torch.abs(torch.mean(quantized) - 1.0).item()
        self.assertLess(mean_diff, 0.08)

        # Test zero tensor
        zeros = torch.zeros(10)
        quantized_zeros = FedPAQ_Client.quantize_tensor(zeros, 4)
        self.assertTrue(torch.equal(quantized_zeros, zeros))


if __name__ == "__main__":
    unittest.main()
