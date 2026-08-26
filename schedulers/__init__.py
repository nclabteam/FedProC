import importlib
from argparse import ArgumentParser
from collections.abc import Callable
from pathlib import Path
from typing import Any

schedulers: dict[str, type[Any]] = {}
optional: dict[str, dict[str, Any]] = {}
compulsory: dict[str, dict[str, Any]] = {}
SCHEDULER_MODES = ("batch", "epoch", "iteration")
args_update_functions: dict[
    str,
    Callable[[ArgumentParser], None] | None,
] = {}

for path in sorted(
    Path(__file__).parent.glob("*.py"), key=lambda item: item.stem.casefold()
):
    name = path.stem
    if name == "__init__":
        continue
    module = importlib.import_module(f".{name}", package=__name__)
    scheduler_type = getattr(module, name, None)
    if not isinstance(scheduler_type, type):
        continue

    schedulers[name] = scheduler_type
    optional[name] = {
        "scheduler_mode": "iteration",
        **getattr(scheduler_type, "optional", {}),
    }
    compulsory[name] = getattr(scheduler_type, "compulsory", {})
    args_update_functions[name] = getattr(scheduler_type, "args_update", None)

globals().update(schedulers)
__all__ = list(schedulers)
SCHEDULERS = list(schedulers)
