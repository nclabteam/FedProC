import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestStrategies(unittest.TestCase):
    def test_fedavg_is_tfl(self):
        from strategies.FedAvg import FedAvg, FedAvg_Client
        from strategies.tFL import tFL, tFL_Client

        self.assertTrue(issubclass(FedAvg, tFL))
        self.assertTrue(issubclass(FedAvg_Client, tFL_Client))

        server = object.__new__(FedAvg)
        self.assertIs(server._client_cls(), FedAvg_Client)

    def test_select_clients_returns_int_ids(self):
        from strategies.FedAvg import FedAvg
        from strategies.tFL import tFL

        server = object.__new__(FedAvg)
        server.num_clients = 3
        server.is_new = {0: False, 1: False, 2: False}
        server.random_join_ratio = False
        server.num_join_clients = 2

        tFL.select_clients(server)
        self.assertEqual(len(server.selected_clients), 2)
        self.assertTrue(all(isinstance(c, int) for c in server.selected_clients))
        self.assertTrue(set(server.selected_clients) <= {0, 1, 2})

    def test_aggregate_client_updates_weighted_average(self):
        from collections import OrderedDict

        import torch

        from strategies.FedAvg import FedAvg

        server = object.__new__(FedAvg)
        server.model = torch.nn.Linear(2, 1, bias=False)
        server.public_model_params = OrderedDict(
            (k, v.detach().clone()) for k, v in server.model.named_parameters()
        )
        name = next(iter(server.public_model_params))
        packages = OrderedDict(
            {
                0: {"score": 1, "regular_model_params": {name: torch.zeros(1, 2)}},
                1: {"score": 3, "regular_model_params": {name: torch.ones(1, 2)}},
            }
        )
        server.aggregate_client_updates(packages)
        # weighted mean = (1*0 + 3*1) / 4 = 0.75
        self.assertTrue(
            torch.allclose(server.public_model_params[name], torch.full((1, 2), 0.75))
        )

    def test_fedbn_is_pfl(self):
        from strategies.FedBN import FedBN, FedBN_Client
        from strategies.pFL import pFL, pFL_Client

        self.assertTrue(issubclass(FedBN, pFL))
        self.assertTrue(issubclass(FedBN_Client, pFL_Client))

    def test_fedpaq_quantize_tensor(self):
        import torch

        from strategies.FedPAQ import FedPAQ_Client

        tensor = torch.ones(100000)
        quantized = FedPAQ_Client.quantize_tensor(tensor, 4)
        mean_diff = torch.abs(torch.mean(quantized) - 1.0).item()
        self.assertLess(mean_diff, 0.08)

        zeros = torch.zeros(10)
        quantized_zeros = FedPAQ_Client.quantize_tensor(zeros, 4)
        self.assertTrue(torch.equal(quantized_zeros, zeros))


if __name__ == "__main__":
    unittest.main()
