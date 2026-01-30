import os
import sys
import unittest
from argparse import Namespace

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock

# Mock framework modules to avoid importing world
for module in ['data_factory', 'losses', 'models', 'optimizers', 'scalers', 'schedulers', 'strategies']:
    mock = MagicMock()
    # Mock the list constants expected by options.py
    setattr(mock, module.upper(), [])
    # Mock the args_update_functions expected by options.py
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


if __name__ == '__main__':
    unittest.main()
