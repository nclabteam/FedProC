import os
import sys
import unittest

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestDataFactory(unittest.TestCase):
    def test_dataset_instantiation(self):
        """Verify that a real dataset class can be instantiated with its config."""
        from data_factory.ETDataset import ETTm1
        from argparse import Namespace
        
        # Mock config using Namespace (dot notation required by BaseDataset)
        configs = Namespace(
            input_len=96,
            output_len=48,
            offset_len=0,
            train_ratio=0.7
        )
        
        dataset = ETTm1(configs)
        
        # Verify core parameters defined in ETDataset.py
        self.assertEqual(dataset.granularity, 15)
        self.assertEqual(dataset.granularity_unit, "minute")
        self.assertIn("HUFL", dataset.column_target)
        self.assertTrue(dataset.url.endswith("ETTm1.csv"))

if __name__ == '__main__':
    unittest.main()
