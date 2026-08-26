import importlib
from pathlib import Path
from typing import Any

augmentations: dict[str, type[Any]] = {}

for path in sorted(Path(__file__).parent.glob("*.py"), key=lambda item: item.stem):
    name = path.stem
    if name == "__init__":
        continue
    module = importlib.import_module(name=f".{name}", package=__name__)
    for class_name, augmentation_type in vars(module).items():
        if (
            isinstance(augmentation_type, type)
            and augmentation_type.__module__ == module.__name__
        ):
            augmentations[class_name] = augmentation_type

globals().update(augmentations)
AUGMENTATIONS = sorted(augmentations)
__all__ = AUGMENTATIONS
