import importlib
import inspect
import os

ATTACKS = []

current_dir = os.path.dirname(__file__)

for filename in os.listdir(current_dir):
    if filename.endswith(".py") and filename != "__init__.py":
        module_name = filename[:-3]
        module = importlib.import_module(f".{module_name}", package=__name__)
        for name, obj in inspect.getmembers(module):
            if (
                inspect.isclass(obj)
                and obj.__module__ == module.__name__
                and module_name == name
            ):
                globals()[name] = obj
                ATTACKS.append(name)

__all__ = ATTACKS
