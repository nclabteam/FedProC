import importlib
from pathlib import Path
from typing import Any

topologies: dict[str, type[Any]] = {}

for path in sorted(Path(__file__).parent.glob("*.py"), key=lambda item: item.stem):
    name = path.stem
    if name in {"__init__", "base"}:
        continue
    module = importlib.import_module(name=f".{name}", package=__name__)
    topology_type = getattr(module, name, None)
    if isinstance(topology_type, type):
        topologies[name] = topology_type

globals().update(topologies)
TOPOLOGIES = list(topologies)
__all__ = TOPOLOGIES
