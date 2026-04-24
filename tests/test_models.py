import os
import sys
import unittest
from argparse import Namespace

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.DLinear import DLinear
from models.FLinear import FLinear
from models.Linear import Linear
from models.UMixer import UMixer
from models.xPatch import xPatch


class TestModels(unittest.TestCase):
    def make_base_config(self, **overrides):
        config = Namespace(
            input_len=16,
            output_len=8,
            input_channels=2,
            output_channels=2,
            moving_avg=3,
            stride=1,
            freq_topk=4,
            rfft=True,
            e_layers=1,
            d_model=8,
            patch_len=4,
            dropout=0.0,
            alpha=0.3,
            beta=0.3,
            padding_patch="end",
            ma_type="EMA",
            learnable=False,
            device="cpu",
        )
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    def assert_forward_shape(self, model, configs):
        batch_size = 2
        x = torch.randn(batch_size, configs.input_len, configs.input_channels)
        output = model(x)
        self.assertEqual(
            output.shape,
            (batch_size, configs.output_len, configs.output_channels),
        )

    def test_linear_forward_shape(self):
        configs = self.make_base_config()
        self.assert_forward_shape(Linear(configs), configs)

    def test_dlinear_forward_shape(self):
        configs = self.make_base_config()
        self.assert_forward_shape(DLinear(configs), configs)

    def test_flinear_forward_shape(self):
        configs = self.make_base_config()
        self.assert_forward_shape(FLinear(configs), configs)

    def test_umixer_forward_shape(self):
        configs = self.make_base_config()
        self.assert_forward_shape(UMixer(configs), configs)

    def test_xpatch_forward_shape(self):
        configs = self.make_base_config()
        self.assert_forward_shape(xPatch(configs), configs)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_flinear_cuda_smoke(self):
        configs = self.make_base_config()
        model = FLinear(configs).to("cuda")
        x = torch.randn(2, configs.input_len, configs.input_channels, device="cuda")
        output = model(x)
        self.assertEqual(
            output.shape,
            (2, configs.output_len, configs.output_channels),
        )


if __name__ == "__main__":
    unittest.main()
