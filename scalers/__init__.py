import importlib
from pathlib import Path
from typing import Any

scalers: dict[str, type[Any]] = {}

for path in sorted(Path(__file__).parent.glob("*.py"), key=lambda item: item.stem):
    name = path.stem
    if name == "__init__":
        continue
    module = importlib.import_module(name=f".{name}", package=__name__)
    scaler_type = getattr(module, name, None)
    if isinstance(scaler_type, type):
        scalers[name] = scaler_type

globals().update(scalers)
SCALERS = list(scalers)
__all__ = SCALERS
