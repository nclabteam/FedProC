import importlib
import os
from typing import Any, Callable, Dict

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Initialize empty dictionaries to store the imported classes, optional dictionaries, and compulsory dictionaries
strategies = {}
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
            strategies[class_name] = class_obj

            # Import optional dictionary
            if hasattr(module, "optional"):
                optional[class_name] = getattr(module, "optional")

            # Import compulsory dictionary
            if hasattr(module, "compulsory"):
                compulsory[class_name] = getattr(module, "compulsory")

            # Import args_update function
            if hasattr(module, "args_update"):
                args_update_functions[class_name] = getattr(module, "args_update")

# Add the imported classes to the module's namespace
globals().update(strategies)


# Function to apply args_update to a parser
def apply_args_update(parser):
    import argparse

    existing_args = {action.dest for action in parser._actions}
    for class_name, update_func in args_update_functions.items():
        # Create a temporary parser to capture the arguments
        temp_parser = argparse.ArgumentParser(add_help=False)
        update_func(temp_parser)

        # Add only the new arguments to the main parser
        for action in temp_parser._actions:
            if action.dest not in existing_args:
                parser._add_action(action)
                existing_args.add(action.dest)


# Optionally, define __all__ to control what gets imported with "from strategies import *"
__all__ = list(strategies.keys())
STRATEGIES = list(strategies.keys())
print(f"{STRATEGIES = }")
