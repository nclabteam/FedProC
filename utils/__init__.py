__all__ = [
    "Decimal",
    "increment_path",
    "Options",
    "SetSeed",
]


def __getattr__(name):
    if name == "Decimal":
        from .decimal import Decimal

        return Decimal
    if name == "increment_path":
        from .general import increment_path

        return increment_path
    if name == "Options":
        from .options import Options

        return Options
    if name == "SetSeed":
        from .seed import SetSeed

        return SetSeed
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
