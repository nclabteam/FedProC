import importlib
import os
from typing import Any, Callable, Dict

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Initialize empty dictionaries to store the imported classes, optional dictionaries, and compulsory dictionaries
schedulers = {}
optional: Dict[Any, dict] = {}
compulsory: Dict[Any, dict] = {}
args_update_functions: Dict[str, Callable] = {}

for filename in os.listdir(current_dir):
    if filename.endswith(".py") and filename != "__init__.py":
        module_name = filename[:-3]  # Remove the .py extension
        module = importlib.import_module(f".{module_name}", package=__package__)

        # Import the main class
        class_name = module_name
        if hasattr(module, class_name):
            class_obj = getattr(module, class_name)
            schedulers[class_name] = class_obj

            # Import optional dictionary
            optional[class_name] = (
                getattr(module, "optional") if hasattr(module, "optional") else {}
            )

            # Import compulsory dictionary
            compulsory[class_name] = (
                getattr(module, "compulsory") if hasattr(module, "compulsory") else {}
            )

            # Import args_update function
            args_update_functions[class_name] = (
                getattr(module, "args_update") if hasattr(module, "args_update") else {}
            )

# Add the imported classes to the module's namespace
globals().update(schedulers)


# Optionally, define __all__ to control what gets imported with "from strategies import *"
__all__ = list(schedulers.keys())
SCHEDULERS = list(schedulers.keys())
