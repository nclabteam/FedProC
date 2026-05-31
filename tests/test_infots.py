import unittest
from argparse import Namespace

import torch

from augs import (cutout, jitter, magnitude_warp, scaling, subsequence,
                  time_warp, window_slice, window_warp)
from models.InfoTS import AutoAUG, InfoTS, TSEncoder


class TestInfoTSAugmentations(unittest.TestCase):
    def setUp(self):
        # Create a mock batch of shape [B, T, D]
        self.B = 4
        self.T = 24
        self.D = 3
        self.x = torch.randn(self.B, self.T, self.D)

    def test_cutout(self):
        aug = cutout(perc=0.15)
        out = aug(self.x)
        self.assertEqual(out.shape, self.x.shape)
        self.assertTrue((out == 0.0).any())

    def test_jitter(self):
        aug = jitter(sigma=0.2)
        out = aug(self.x)
        self.assertEqual(out.shape, self.x.shape)
        self.assertFalse(torch.equal(out, self.x))

    def test_scaling(self):
        aug = scaling(sigma=0.3)
        out = aug(self.x)
        self.assertEqual(out.shape, self.x.shape)
        self.assertFalse(torch.equal(out, self.x))

    def test_time_warp(self):
        aug = time_warp(n_speed_change=3, max_speed_ratio=1.5)
        out = aug(self.x)
        self.assertEqual(out.shape, self.x.shape)

    def test_magnitude_warp(self):
        aug = magnitude_warp(n_speed_change=3, max_speed_ratio=1.5)
        out = aug(self.x)
        self.assertEqual(out.shape, self.x.shape)

    def test_window_slice(self):
        aug = window_slice(reduce_ratio=0.6)
        out = aug(self.x)
        self.assertEqual(out.shape, self.x.shape)

    def test_window_warp(self):
        aug = window_warp(window_ratio=0.3, scales=[0.6, 1.8])
        out = aug(self.x)
        self.assertEqual(out.shape, self.x.shape)

    def test_subsequence(self):
        aug = subsequence()
        out = aug(self.x)
        self.assertEqual(out.shape, self.x.shape)


class TestInfoTSModel(unittest.TestCase):
    def test_auto_aug(self):
        aug = AutoAUG(aug_p1=1.0)
        x = torch.randn(4, 16, 2)
        aug1, aug2 = aug(x)
        self.assertEqual(aug1.shape, x.shape)
        self.assertEqual(aug2.shape, x.shape)

    def test_infots_model_flow(self):
        configs = Namespace(
            input_channels=2,
            output_channels=2,
            output_len=8,
            infots_repr_dim=64,
            infots_hidden_dim=16,
            infots_depth=3,
            infots_beta=1.0,
            infots_meta_beta=1.0,
            infots_aug_p1=0.2,
            infots_aug_p2=0.0,
            infots_k=4,
            batch_size=4,
        )

        model = InfoTS(configs)
        x = torch.randn(configs.batch_size, 32, configs.input_channels)

        # Test pretrain_loss
        loss = model.pretrain_loss(x)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)

        # Test meta_step
        meta_opt = torch.optim.AdamW(model.aug.parameters(), lr=0.01)
        meta_head_opt = torch.optim.AdamW(model.meta_unsup_head.parameters(), lr=0.01)
        meta_loss = model.meta_step(x, meta_opt, meta_head_opt)
        self.assertIsInstance(meta_loss, float)

        # Test forward pass (forecasting output shape)
        out = model(x)
        self.assertEqual(
            out.shape, (configs.batch_size, configs.output_len, configs.output_channels)
        )


if __name__ == "__main__":
    unittest.main()
