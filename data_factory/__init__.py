import importlib
import inspect
import os

# Initialize a list to store all imported class names
DATASETS = []

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
                and name != "BaseDataset"
            ):
                # Add the class to the current module's namespace
                globals()[name] = obj
                # Add the class name to the list
                DATASETS.append(name)

__all__ = DATASETS
print(f"{DATASETS = }")
