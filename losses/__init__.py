import importlib
import inspect
import os
import sys

# Initialize lists to store imported class names
LOSSES = []  # For training losses (eval_only=False or not set)
EVAL_LOSSES = []  # For evaluation-only losses (eval_only=True)

# Get the current directory of this __init__.py
current_dir = os.path.dirname(__file__)

# Loop through all files in the directory
for filename in os.listdir(current_dir):
    # Only consider .py files that are not __init__.py
    if filename.endswith(".py") and filename != "__init__.py":
        # Get the module name by stripping .py
        module_name = filename[:-3]
        # Dynamically import the module
        module = importlib.import_module(f".{module_name}", package=__name__)

        # Loop through all members of the module
        for name, obj in inspect.getmembers(module):
            # Check if the member is a class and is defined in this module
            if (
                inspect.isclass(obj)
                and obj.__module__ == module.__name__
                and module_name == name
            ):
                # Check if the class has eval_only attribute
                eval_only = getattr(obj, "eval_only", False)
                # Add the class to the current module's namespace
                globals()[name] = obj

                if eval_only:
                    # Add to evaluation-only losses
                    EVAL_LOSSES.append(name)
                else:
                    # Add to training losses
                    LOSSES.append(name)

__all__ = LOSSES + EVAL_LOSSES


def evaluation_result(y_pred, y_true):
    results = {}

    # Include training losses
    for loss in LOSSES:
        results[loss] = getattr(sys.modules[__name__], loss)()(y_true, y_pred).item()

    # Include evaluation-only losses
    for loss in EVAL_LOSSES:
        results[loss] = getattr(sys.modules[__name__], loss)()(y_true, y_pred).item()

    return results
