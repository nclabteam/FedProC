import importlib
import os
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
MODELS = list(_MODULE_NAMES)
__all__ = list(MODELS)
_MODULE_CACHE: Dict[str, Any] = {}


def _load_module(module_name: str):
    module = _MODULE_CACHE.get(module_name)
    if module is None:
        module = importlib.import_module(f".{module_name}", package=__name__)
        _MODULE_CACHE[module_name] = module
    return module


def _load_model_class(model_name: str):
    if model_name not in MODELS:
        raise AttributeError(f"module {__name__!r} has no attribute {model_name!r}")
    module = _load_module(model_name)
    if not hasattr(module, model_name):
        raise AttributeError(f"module {module.__name__!r} has no class {model_name!r}")
    return getattr(module, model_name)


class _LazyModuleRegistry(Mapping):
    def __init__(self, attribute_name: str, default: Any):
        self.attribute_name = attribute_name
        self.default = default

    def _resolve(self, key: str, fallback: Any) -> Any:
        cls = _load_model_class(key)
        return getattr(cls, self.attribute_name, fallback)

    def __getitem__(self, key):
        if key not in MODELS:
            raise KeyError(key)
        return self._resolve(key, self.default)

    def __iter__(self):
        return iter(MODELS)

    def __len__(self):
        return len(MODELS)

    def get(self, key, default=None):
        if key not in MODELS:
            return default
        return self._resolve(key, self.default if default is None else default)


optional: Mapping[Any, dict] = _LazyModuleRegistry("optional", {})
compulsory: Mapping[Any, dict] = _LazyModuleRegistry("compulsory", {})
args_update_functions: Mapping[str, Callable] = _LazyModuleRegistry("args_update", None)


def __getattr__(name: str):
    return _load_model_class(name)
