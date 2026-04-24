import ast
import importlib
import os
from typing import Any, Dict


def _discover_dataset_modules() -> list[str]:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return sorted(
        filename[:-3]
        for filename in os.listdir(current_dir)
        if filename.endswith(".py") and filename != "__init__.py"
    )


_MODULE_NAMES = _discover_dataset_modules()
_MODULE_CACHE: Dict[str, Any] = {}
_DATASET_TO_MODULE: Dict[str, str] = {}


def _load_module(module_name: str):
    module = _MODULE_CACHE.get(module_name)
    if module is None:
        module = importlib.import_module(f".{module_name}", package=__name__)
        _MODULE_CACHE[module_name] = module
    return module


def _discover_dataset_names() -> list[str]:
    dataset_names = []
    for module_name in _MODULE_NAMES:
        module_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"{module_name}.py",
        )
        with open(module_path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=module_path)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name != "BaseDataset":
                _DATASET_TO_MODULE[node.name] = module_name
                dataset_names.append(node.name)
    return sorted(dataset_names)


DATASETS = _discover_dataset_names()
__all__ = list(DATASETS)


def __getattr__(name: str):
    module_name = _DATASET_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _load_module(module_name)
    return getattr(module, name)
