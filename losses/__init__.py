import importlib
from pathlib import Path

from torch import Tensor, nn

LOSSES: list[str] = []
EVAL_LOSSES: list[str] = []
CONTEXT_LOSSES: list[str] = []

for path in sorted(
    Path(__file__).parent.glob("*.py"), key=lambda item: item.stem.casefold()
):
    name = path.stem
    if name == "__init__":
        continue
    module = importlib.import_module(f".{name}", package=__name__)
    loss_type = getattr(module, name, None)
    if not isinstance(loss_type, type) or not issubclass(loss_type, nn.Module):
        continue

    globals()[name] = loss_type
    if getattr(loss_type, "context_only", False):
        CONTEXT_LOSSES.append(name)
    elif getattr(loss_type, "eval_only", False):
        EVAL_LOSSES.append(name)
    else:
        LOSSES.append(name)

__all__ = [*LOSSES, *EVAL_LOSSES, *CONTEXT_LOSSES, "evaluation_result"]
_EVALUATORS: tuple[tuple[str, nn.Module], ...] = tuple(
    (name, globals()[name]())
    for name in (*LOSSES, *EVAL_LOSSES)
    if getattr(globals()[name], "generic_eval", True)
)


def evaluation_result(y_pred: Tensor, y_true: Tensor) -> dict[str, float]:
    """Evaluate metrics that only require predictions and observations."""
    return {
        name: evaluator(input=y_pred, target=y_true).item()
        for name, evaluator in _EVALUATORS
    }
