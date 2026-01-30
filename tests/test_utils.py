import os
import sys
import unittest
from argparse import Namespace

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock

def mock_framework():
    # Mock framework modules to avoid importing the whole world during light tests
    for module in ['data_factory', 'losses', 'models', 'optimizers', 'scalers', 'schedulers', 'strategies']:
        mock = MagicMock()
        setattr(mock, module.upper(), [])
        setattr(mock, 'args_update_functions', {})
        sys.modules[module] = mock

from utils.seed import SetSeed

class TestUtils(unittest.TestCase):
    def test_seed_setting(self):
        """Verify that seed setting is reproducible."""
        SetSeed(42).set()
        import torch
        import numpy as np
        import random
        
        a1 = torch.randn(5)
        n1 = np.random.randn(5)
        r1 = random.random()
        
        SetSeed(42).set()
        a2 = torch.randn(5)
        n2 = np.random.randn(5)
        r2 = random.random()
        
        torch.testing.assert_close(a1, a2)
        np.testing.assert_array_almost_equal(n1, n2)
        self.assertEqual(r1, r2)

class TestModels(unittest.TestCase):
    def test_dlinear_forward(self):
        """Verify DLinear model forward pass with expected shapes."""
        import torch
        from models.DLinear import DLinear
        
        # Mock configs
        configs = MagicMock()
        configs.input_len = 96
        configs.output_len = 48
        configs.moving_avg = 25
        configs.stride = 1
        
        model = DLinear(configs)
        batch_size, channels = 4, 7
        x = torch.randn(batch_size, configs.input_len, channels)
        output = model(x)
        
        self.assertEqual(output.shape, (batch_size, configs.output_len, channels))

class TestDataCharacteristics(unittest.TestCase):
    def test_transition_value_logic(self):
        """Verify transition value computation logic for time series."""
        import polars as pl
        import numpy as np
        from data_factory.base import TimeSeriesCharacteristics
        
        # Create a simple predictable signal
        data = pl.DataFrame({
            "v1": np.sin(np.linspace(0, 10, 100)),
            "v2": np.random.randn(100)
        })
        
        result = TimeSeriesCharacteristics.get_transition_value(data)
        self.assertIn("v1", result.columns)
        self.assertIn("v2", result.columns)
        self.assertEqual(result.shape, (1, 3)) # variable + v1 + v2
        self.assertEqual(result["variable"][0], "transition")

if __name__ == '__main__':
    unittest.main()
