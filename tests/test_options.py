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


if __name__ == "__main__":
    unittest.main()
