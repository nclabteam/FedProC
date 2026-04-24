import os
import sys
import tempfile
import unittest
from argparse import Namespace

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Linear
from strategies.base import SharedMethods


class TestCheckpointing(unittest.TestCase):
    def make_configs(self):
        return Namespace(
            model="Linear",
            input_len=8,
            output_len=4,
        )

    def make_model(self):
        configs = self.make_configs()
        model = Linear(configs=configs)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.copy_(torch.randn_like(parameter))
        return model, configs

    def test_safe_checkpoint_round_trip(self):
        model, configs = self.make_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            SharedMethods.save_model(
                model=model,
                path=tmpdir,
                name="server",
                postfix="best",
                configs=configs,
                metadata={"save_type": "best"},
            )
            checkpoint_path = os.path.join(tmpdir, "server_best.pt")

            payload = torch.load(checkpoint_path, weights_only=True)
            self.assertEqual(payload["format"], SharedMethods.checkpoint_format)
            self.assertEqual(payload["model_name"], "Linear")
            self.assertIn("state_dict", payload)

            restored = SharedMethods.load_checkpoint_model(checkpoint_path)
            self.assertIsInstance(restored, Linear)
            for saved_param, restored_param in zip(
                model.state_dict().values(), restored.state_dict().values()
            ):
                self.assertTrue(torch.equal(saved_param.cpu(), restored_param.cpu()))

    def test_legacy_checkpoint_requires_explicit_opt_in(self):
        model, _ = self.make_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "legacy.pt")
            torch.save(model, checkpoint_path)

            with self.assertRaises(ValueError):
                SharedMethods.load_checkpoint_model(checkpoint_path)

            restored = SharedMethods.load_checkpoint_model(
                checkpoint_path,
                allow_unsafe_legacy=True,
            )
            self.assertIsInstance(restored, Linear)


if __name__ == "__main__":
    unittest.main()
