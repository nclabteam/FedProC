import importlib
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestLazyRegistries(unittest.TestCase):
    @staticmethod
    def restore_modules(module_name: str, saved_modules: dict) -> None:
        """Restore a package tree replaced by a registry reload test."""
        for loaded_name in list(sys.modules):
            if loaded_name == module_name or loaded_name.startswith(f"{module_name}."):
                sys.modules.pop(loaded_name)
        sys.modules.update(saved_modules)

    def reload_module(self, module_name: str):
        saved_modules = {
            loaded_name: module
            for loaded_name, module in sys.modules.items()
            if loaded_name == module_name or loaded_name.startswith(f"{module_name}.")
        }
        for loaded_name in list(sys.modules):
            if loaded_name == module_name or loaded_name.startswith(f"{module_name}."):
                sys.modules.pop(loaded_name)
        self.addCleanup(self.restore_modules, module_name, saved_modules)
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
        fake_parent = type("Parent", (), {"optional": {"inherited": 1}})
        fake_cls = type(
            "FLinear",
            (fake_parent,),
            {
                "args_update": fake_update,
                "optional": {"local": 2},
            },
        )
        with patch.object(
            models,
            "_load_module",
            return_value=SimpleNamespace(FLinear=fake_cls),
        ):
            update_func = models.args_update_functions["FLinear"]
            optional = models.optional["FLinear"]

        self.assertTrue(callable(update_func))
        self.assertEqual(optional, {"inherited": 1, "local": 2})
        self.assertNotIn("models.GPT4TS", sys.modules)

    def test_strategies_registry_is_lazy(self):
        strategies = self.reload_module("strategies")

        self.assertIn("LocalOnly", strategies.STRATEGIES)
        self.assertNotIn("base", strategies.STRATEGIES)
        self.assertNotIn("strategies.FedTrend", sys.modules)

        strategies._load_strategy_class(strategy_name="DLSA")
        self.assertIsInstance(strategies.FedRidge, type)

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
        self.assertTrue(
            {
                "BaseDataset",
                "CustomDataset",
                "CustomOnSingleDataset",
                "DataFrameOptimizer",
                "FileManager",
                "TimeSeriesCharacteristics",
            }.isdisjoint(data_factory.DATASETS)
        )
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
