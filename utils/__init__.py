from .decimal import Decimal
from .general import increment_path
from .seed import SetSeed

__all__ = [
    "Decimal",
    "increment_path",
    "ModelSummarizer",
    "Options",
    "SetSeed",
]


def __getattr__(name):
    if name == "ModelSummarizer":
        from .model_info import ModelSummarizer

        return ModelSummarizer
    if name == "Options":
        from .options import Options

        return Options
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
