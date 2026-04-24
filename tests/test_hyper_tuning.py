import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.hyper_tuning import BASE_ARGS, HYPERPARAMS, build_command, generate_combinations


class TestHyperTuning(unittest.TestCase):
    def test_base_args_use_valid_cli_values(self):
        self.assertIn("--scaler=Standard", BASE_ARGS)
        self.assertNotIn("--scaler=StandardScaler", BASE_ARGS)

    def test_build_command_uses_python_and_valid_run_name(self):
        param_keys = ["strategy", "iterations", "model"]
        param_values = ("LocalOnly", 100, "Linear")

        command, run_name = build_command(param_keys, param_values)

        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], "main.py")
        self.assertIn("--strategy=LocalOnly", command)
        self.assertIn("--iterations=100", command)
        self.assertIn("--model=Linear", command)
        self.assertIn(f"--name={run_name}", command)
        self.assertEqual(run_name, "strategyLocalOnly_iterations100_modelLinear")

    def test_combination_count_matches_hyperparams(self):
        _, combos = generate_combinations(HYPERPARAMS)
        expected = 1
        for values in HYPERPARAMS.values():
            expected *= len(values)
        self.assertEqual(len(combos), expected)


if __name__ == "__main__":
    unittest.main()
