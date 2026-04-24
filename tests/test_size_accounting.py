import os
import sys
import unittest

import torch
from torch.utils.data import TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.base import SharedMethods


class TinyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2, bias=True)
        self.register_buffer("scale", torch.ones(3, dtype=torch.float32))


class TestSizeAccounting(unittest.TestCase):
    def test_tensor_size_matches_element_count(self):
        tensor = torch.ones((2, 3), dtype=torch.float32)
        expected_mb = tensor.element_size() * tensor.nelement() / (1024**2)
        self.assertEqual(SharedMethods.get_size(tensor), expected_mb)

    def test_nested_dict_and_list_size_sums_children_once(self):
        tensor_a = torch.ones((2, 2), dtype=torch.float32)
        tensor_b = torch.ones((1, 4), dtype=torch.float32)
        payload = {"a": tensor_a, "nested": [tensor_b]}

        expected_bytes = (
            tensor_a.element_size() * tensor_a.nelement()
            + tensor_b.element_size() * tensor_b.nelement()
        )
        expected_mb = expected_bytes / (1024**2)
        self.assertEqual(SharedMethods.get_size(payload), expected_mb)

    def test_tensor_dataset_size_matches_sum_of_tensors(self):
        x = torch.ones((3, 4), dtype=torch.float32)
        y = torch.ones((3, 2), dtype=torch.float32)
        dataset = TensorDataset(x, y)
        expected_mb = (
            x.element_size() * x.nelement() + y.element_size() * y.nelement()
        ) / (1024**2)
        self.assertEqual(SharedMethods.get_size(dataset), expected_mb)

    def test_module_size_includes_parameters_and_buffers(self):
        module = TinyModule()
        expected_bytes = sum(
            parameter.element_size() * parameter.nelement()
            for parameter in module.parameters()
        )
        expected_bytes += sum(
            buffer.element_size() * buffer.nelement() for buffer in module.buffers()
        )
        expected_mb = expected_bytes / (1024**2)
        self.assertEqual(SharedMethods.get_size(module), expected_mb)


if __name__ == "__main__":
    unittest.main()
