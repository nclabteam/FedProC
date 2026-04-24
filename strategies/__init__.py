import importlib
import os
import sys
from collections.abc import Mapping
from typing import Any, Callable, Dict


def _discover_module_names() -> list[str]:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return sorted(
        filename[:-3]
        for filename in os.listdir(current_dir)
        if filename.endswith(".py") and filename != "__init__.py"
    )


_MODULE_NAMES = _discover_module_names()
STRATEGIES = list(_MODULE_NAMES)
__all__ = list(STRATEGIES)
_MODULE_CACHE: Dict[str, Any] = {}


def _load_module(module_name: str):
    module = _MODULE_CACHE.get(module_name)
    if module is None:
        module = importlib.import_module(f".{module_name}", package=__name__)
        _MODULE_CACHE[module_name] = module
        # importlib.import_module binds the submodule into the parent package's
        # __dict__, which would cause getattr(strategies, name) to return the
        # module object instead of the class, bypassing __getattr__.
        vars(sys.modules[__name__]).pop(module_name, None)
    return module


def _load_strategy_class(strategy_name: str):
    if strategy_name not in STRATEGIES:
        raise AttributeError(f"module {__name__!r} has no attribute {strategy_name!r}")
    module = _load_module(strategy_name)
    if not hasattr(module, strategy_name):
        raise AttributeError(
            f"module {module.__name__!r} has no class {strategy_name!r}"
        )
    return getattr(module, strategy_name)


class _LazyModuleRegistry(Mapping):
    def __init__(self, attribute_name: str, default: Any):
        self.attribute_name = attribute_name
        self.default = default

    def __getitem__(self, key):
        if key not in STRATEGIES:
            raise KeyError(key)
        module = _load_module(key)
        return getattr(module, self.attribute_name, self.default)

    def __iter__(self):
        return iter(STRATEGIES)

    def __len__(self):
        return len(STRATEGIES)

    def get(self, key, default=None):
        if key not in STRATEGIES:
            return default
        module = _load_module(key)
        return getattr(
            module,
            self.attribute_name,
            self.default if default is None else default,
        )


optional: Mapping[Any, dict] = _LazyModuleRegistry("optional", {})
compulsory: Mapping[Any, dict] = _LazyModuleRegistry("compulsory", {})
args_update_functions: Mapping[str, Callable] = _LazyModuleRegistry(
    "args_update", {}
)


def __getattr__(name: str):
    return _load_strategy_class(name)
