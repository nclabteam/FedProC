import os
import sys
import unittest
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.tFL import tFL_Client as Client
from strategies.base import SharedMethods


class TrackableModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
        self.devices = []

    def forward(self, x, **kwargs):
        return self.layer(x)

    def to(self, *args, **kwargs):
        if args:
            self.devices.append(str(args[0]))
        return super().to(*args, **kwargs)


class WrappedLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.original_layer = torch.nn.Linear(1, 1)
        self.lora_A = torch.nn.Parameter(torch.ones(1, 1))
        self.lora_B = torch.nn.Parameter(torch.ones(1, 1))

    def forward(self, x):
        return self.original_layer(x) + x @ self.lora_A @ self.lora_B


class TestDevicePolicy(unittest.TestCase):
    def make_loader(self):
        x = torch.ones(2, 1)
        y = torch.ones(2, 1)
        x_mark = torch.ones(2, 1)
        y_mark = torch.ones(2, 1)
        return DataLoader(TensorDataset(x, y, x_mark, y_mark), batch_size=1)

    def test_calculate_loss_offloads_when_requested(self):
        model = TrackableModel()
        SharedMethods.calculate_loss(
            model=model,
            dataloader=self.make_loader(),
            criterion=torch.nn.MSELoss(),
            device="cpu",
            offload_after=True,
        )
        self.assertEqual(model.devices[-1], "cpu")

    def test_calculate_loss_can_keep_resident(self):
        model = TrackableModel()
        SharedMethods.calculate_loss(
            model=model,
            dataloader=self.make_loader(),
            criterion=torch.nn.MSELoss(),
            device="cpu",
            offload_after=False,
        )
        self.assertEqual(model.devices, ["cpu"])

    def test_train_one_epoch_offloads_when_requested(self):
        model = TrackableModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        SharedMethods.train_one_epoch(
            model=model,
            dataloader=self.make_loader(),
            optimizer=optimizer,
            criterion=torch.nn.MSELoss(),
            scheduler=scheduler,
            device="cpu",
            offload_after=True,
        )
        self.assertEqual(model.devices[-1], "cpu")

    def test_train_one_epoch_can_keep_model_resident(self):
        model = TrackableModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        SharedMethods.train_one_epoch(
            model=model,
            dataloader=self.make_loader(),
            optimizer=optimizer,
            criterion=torch.nn.MSELoss(),
            scheduler=scheduler,
            device="cpu",
            offload_after=False,
        )
        self.assertEqual(model.devices, ["cpu"])

    def make_fake_client(self, efficiency):
        class FakeTrainModel:
            def __init__(self):
                self.devices = []

            def to(self, device):
                self.devices.append(str(device))
                return self

        class FakeClient:
            pass

        client = FakeClient()
        client.efficiency = efficiency
        client.epochs = 3
        client.parallel = False
        client.model = FakeTrainModel()
        client.optimizer = object()
        client.loss = object()
        client.scheduler = SimpleNamespace(get_last_lr=lambda: [0.1])
        client.device = "cpu"
        client.metrics = {"train_time": [], "lr": []}
        client.load_train_data = lambda: "loader"
        client.calls = []

        def train_one_epoch(**kwargs):
            client.calls.append(kwargs["offload_after"])

        client.train_one_epoch = train_one_epoch
        return client

    def test_client_train_low_offloads_each_epoch(self):
        client = self.make_fake_client("low")
        Client.train(client)
        self.assertEqual(client.calls, [True, True, True])
        self.assertEqual(client.model.devices, [])

    def test_client_train_med_offloads_once_after_epochs(self):
        client = self.make_fake_client("med")
        Client.train(client)
        self.assertEqual(client.calls, [False, False, False])
        self.assertEqual(client.model.devices, ["cpu"])

    def test_client_train_high_keeps_model_resident(self):
        client = self.make_fake_client("high")
        Client.train(client)
        self.assertEqual(client.calls, [False, False, False])
        self.assertEqual(client.model.devices, [])

    def test_clone_model_to_cpu_preserves_wrapped_modules(self):
        client = type("ClientStub", (), {})()
        wrapped = WrappedLinear()
        with torch.no_grad():
            wrapped.original_layer.weight.fill_(2.0)
            wrapped.original_layer.bias.fill_(3.0)
            wrapped.lora_A.fill_(4.0)
            wrapped.lora_B.fill_(5.0)

        clone = Client._clone_model_to_cpu(client, wrapped)

        self.assertIsNot(clone, wrapped)
        self.assertIsInstance(clone, WrappedLinear)
        self.assertEqual(clone.original_layer.weight.device.type, "cpu")
        self.assertTrue(torch.equal(clone.lora_A, wrapped.lora_A.cpu()))
        self.assertTrue(torch.equal(clone.lora_B, wrapped.lora_B.cpu()))
        with torch.no_grad():
            clone.original_layer.weight.fill_(9.0)
        self.assertFalse(
            torch.equal(clone.original_layer.weight, wrapped.original_layer.weight)
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_calculate_loss_cuda_smoke(self):
        model = TrackableModel().to("cuda")
        losses = SharedMethods.calculate_loss(
            model=model,
            dataloader=self.make_loader(),
            criterion=torch.nn.MSELoss(),
            device="cuda",
            offload_after=True,
        )
        self.assertEqual(len(losses), 2)


if __name__ == "__main__":
    unittest.main()
