import ast
import importlib
import os
import sys
from collections.abc import Mapping
from typing import Any, Callable, Dict


def _discover_strategy_map() -> Dict[str, str]:
    """Scan strategy files via AST and return {class_name: module_stem}.

    A file named Foo.py registers every top-level class whose name ends with
    'Foo' (e.g. Foo, DFoo, XFoo).  This lets co-located variants like
    DFedProx live inside FedProx.py without a separate file.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    result: Dict[str, str] = {}
    for filename in sorted(os.listdir(current_dir)):
        if not filename.endswith(".py") or filename == "__init__.py":
            continue
        stem = filename[:-3]
        filepath = os.path.join(current_dir, filename)
        try:
            with open(filepath, encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=filepath)
            classes = [
                n.name
                for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name.endswith(stem)
            ]
        except SyntaxError:
            classes = []
        for cls in classes or [stem]:
            result[cls] = stem
    return result


_STRATEGY_MAP = _discover_strategy_map()  # {class_name: module_stem}
STRATEGIES = sorted(_STRATEGY_MAP)
__all__ = list(STRATEGIES)
_MODULE_CACHE: Dict[str, Any] = {}


def _load_module(strategy_name: str):
    module_name = _STRATEGY_MAP[strategy_name]
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

    def _resolve(self, key: str, fallback: Any) -> Any:
        cls = _load_strategy_class(key)
        return getattr(cls, self.attribute_name, fallback)

    def __getitem__(self, key):
        if key not in STRATEGIES:
            raise KeyError(key)
        return self._resolve(key, self.default)

    def __iter__(self):
        return iter(STRATEGIES)

    def __len__(self):
        return len(STRATEGIES)

    def get(self, key, default=None):
        if key not in STRATEGIES:
            return default
        return self._resolve(key, self.default if default is None else default)


optional: Mapping[Any, dict] = _LazyModuleRegistry("optional", {})
compulsory: Mapping[Any, dict] = _LazyModuleRegistry("compulsory", {})
args_update_functions: Mapping[str, Callable] = _LazyModuleRegistry("args_update", None)


def __getattr__(name: str):
    return _load_strategy_class(name)
