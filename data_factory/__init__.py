import ast
import importlib
import os
from types import ModuleType
from typing import Dict


def _discover_dataset_modules() -> list[str]:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return sorted(
        filename[:-3]
        for filename in os.listdir(current_dir)
        if filename.endswith(".py") and filename != "__init__.py"
    )


_MODULE_NAMES = _discover_dataset_modules()
_MODULE_CACHE: Dict[str, ModuleType] = {}
_DATASET_TO_MODULE: Dict[str, str] = {}


def _load_module(module_name: str) -> ModuleType:
    module = _MODULE_CACHE.get(module_name)
    if module is None:
        module = importlib.import_module(f".{module_name}", package=__name__)
        _MODULE_CACHE[module_name] = module
        # Importing a submodule makes Python auto-bind it onto this package
        # under its own name (e.g. `data_factory.M4` = the M4 module object).
        # When a dataset class shares that exact name (M4.py's `M4` class,
        # ThreeW.py's `ThreeW` class), that auto-bind permanently shadows the
        # class for every subsequent `from data_factory import <name>` in the
        # process, since __getattr__ below only fires when the attribute is
        # still missing. Re-assert the real class bindings immediately so the
        # shadowing never sticks.
        for dataset_name, owning_module in _DATASET_TO_MODULE.items():
            if owning_module == module_name:
                globals()[dataset_name] = getattr(module, dataset_name)
    return module


def _discover_dataset_names() -> list[str]:
    classes = {}
    for module_name in _MODULE_NAMES:
        module_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"{module_name}.py",
        )
        with open(module_path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=module_path)
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes[node.name] = (
                    module_name,
                    {
                        base.id if isinstance(base, ast.Name) else base.attr
                        for base in node.bases
                        if isinstance(base, (ast.Name, ast.Attribute))
                    },
                )

    dataset_names = {"BaseDataset"}
    while subclasses := {
        name
        for name, (_, bases) in classes.items()
        if name not in dataset_names and bases & dataset_names
    }:
        dataset_names.update(subclasses)

    base_module = classes["BaseDataset"][0]
    for dataset_name in dataset_names:
        module_name = classes[dataset_name][0]
        if module_name != base_module and not dataset_name.startswith("_"):
            _DATASET_TO_MODULE[dataset_name] = module_name
    return sorted(_DATASET_TO_MODULE)


DATASETS = _discover_dataset_names()
__all__ = list(DATASETS)


def __getattr__(name: str) -> type:
    module_name = _DATASET_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _load_module(module_name=module_name)
    return getattr(module, name)
