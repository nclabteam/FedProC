import os
import sys
import unittest

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestModels(unittest.TestCase):
    def test_dlinear_forward(self):
        """Verify DLinear model forward pass with expected shapes."""
        import torch
        from unittest.mock import MagicMock
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

if __name__ == '__main__':
    unittest.main()
