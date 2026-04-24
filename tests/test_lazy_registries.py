import importlib
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestLazyRegistries(unittest.TestCase):
    def reload_module(self, module_name: str):
        for loaded_name in list(sys.modules):
            if loaded_name == module_name or loaded_name.startswith(f"{module_name}."):
                sys.modules.pop(loaded_name)
        return importlib.import_module(module_name)

    def test_models_registry_is_lazy(self):
        models = self.reload_module("models")

        self.assertIn("Linear", models.MODELS)
        self.assertNotIn("models.GPT4TS", sys.modules)

        fake_linear = type("Linear", (), {})
        with patch.object(
            models,
            "_load_module",
            return_value=SimpleNamespace(Linear=fake_linear),
        ):
            linear_cls = models.Linear

        self.assertEqual(linear_cls.__name__, "Linear")
        self.assertNotIn("models.GPT4TS", sys.modules)

    def test_model_args_update_mapping_loads_only_requested_module(self):
        models = self.reload_module("models")

        fake_update = lambda parser: parser
        with patch.object(
            models,
            "_load_module",
            return_value=SimpleNamespace(args_update=fake_update),
        ):
            update_func = models.args_update_functions["FLinear"]

        self.assertTrue(callable(update_func))
        self.assertNotIn("models.GPT4TS", sys.modules)

    def test_strategies_registry_is_lazy(self):
        strategies = self.reload_module("strategies")

        self.assertIn("LocalOnly", strategies.STRATEGIES)
        self.assertNotIn("strategies.FedTrend", sys.modules)

        fake_strategy = type("LocalOnly", (), {})
        with patch.object(
            strategies,
            "_load_module",
            return_value=SimpleNamespace(LocalOnly=fake_strategy),
        ):
            strategy_cls = strategies.LocalOnly

        self.assertEqual(strategy_cls.__name__, "LocalOnly")
        self.assertNotIn("strategies.FedTrend", sys.modules)

    def test_data_factory_registry_is_lazy(self):
        data_factory = self.reload_module("data_factory")

        self.assertIn("ETDatasetHour", data_factory.DATASETS)
        self.assertNotIn("data_factory.ETDataset", sys.modules)

        fake_dataset = type("ETDatasetHour", (), {})
        with patch.object(
            data_factory,
            "_load_module",
            return_value=SimpleNamespace(ETDatasetHour=fake_dataset),
        ):
            dataset_cls = data_factory.ETDatasetHour

        self.assertEqual(dataset_cls.__name__, "ETDatasetHour")


if __name__ == "__main__":
    unittest.main()
