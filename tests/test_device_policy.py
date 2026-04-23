import os
import sys
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.base import SharedMethods


class TrackableModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
        self.devices = []

    def forward(self, x):
        return self.layer(x)

    def to(self, *args, **kwargs):
        if args:
            self.devices.append(str(args[0]))
        return super().to(*args, **kwargs)


class TestDevicePolicy(unittest.TestCase):
    def make_loader(self):
        x = torch.ones(2, 1)
        y = torch.ones(2, 1)
        return DataLoader(TensorDataset(x, y), batch_size=1)

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


if __name__ == "__main__":
    unittest.main()
