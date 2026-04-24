import os
import sys
import unittest
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.options import Options


class TestOptions(unittest.TestCase):
    def parse_efficiency(self, value):
        with patch.object(
            sys,
            "argv",
            [
                "main.py",
                "--efficiency",
                value,
                "--project",
                "runs_test_options",
            ],
        ):
            return Options(root=".").parse_options().args.efficiency

    def test_efficiency_choices(self):
        for value in ["low", "med", "high"]:
            self.assertEqual(self.parse_efficiency(value), value)

    def test_efficiency_defaults_to_high(self):
        with patch.object(sys, "argv", ["main.py", "--project", "runs_test_options"]):
            self.assertEqual(Options(root=".").parse_options().args.efficiency, "high")

    def test_invalid_efficiency_rejected(self):
        with patch.object(sys, "argv", ["main.py", "--efficiency", "max"]):
            with self.assertRaises(SystemExit):
                Options(root=".").parse_options()

    def test_cpu_device_clears_device_ids(self):
        with patch.object(
            sys,
            "argv",
            [
                "main.py",
                "--device",
                "cpu",
                "--device_id",
                "0,1",
                "--project",
                "runs_test_options",
            ],
        ):
            options = Options(root=".").parse_options()
            options.fix_args()
            self.assertEqual(options.args.device, "cpu")
            self.assertEqual(options.args.device_id, "")

    def test_missing_cuda_downgrades_to_cpu_and_clears_device_ids(self):
        with patch("torch.cuda.is_available", return_value=False):
            with patch.object(
                sys,
                "argv",
                [
                    "main.py",
                    "--device",
                    "cuda",
                    "--device_id",
                    "0",
                    "--project",
                    "runs_test_options",
                ],
            ):
                options = Options(root=".").parse_options()
                options.fix_args()
                self.assertEqual(options.args.device, "cpu")
                self.assertEqual(options.args.device_id, "")

    def test_invalid_cuda_device_id_rejected(self):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=1):
                with patch.object(
                    sys,
                    "argv",
                    [
                        "main.py",
                        "--device",
                        "cuda",
                        "--device_id",
                        "2",
                        "--project",
                        "runs_test_options",
                    ],
                ):
                    options = Options(root=".").parse_options()
                    with self.assertRaises(ValueError):
                        options.fix_args()

    def test_valid_cuda_device_ids_are_normalized(self):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=2):
                with patch.object(
                    sys,
                    "argv",
                    [
                        "main.py",
                        "--device",
                        "cuda",
                        "--device_id",
                        " 0, 1 ",
                        "--project",
                        "runs_test_options",
                    ],
                ):
                    options = Options(root=".").parse_options()
                    options.fix_args()
                    self.assertEqual(options.args.device, "cuda")
                    self.assertEqual(options.args.device_id, "0,1")


if __name__ == "__main__":
    unittest.main()
