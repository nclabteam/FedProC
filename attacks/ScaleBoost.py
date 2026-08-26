from collections import OrderedDict
from typing import Any

from .base import Attack


class ScaleBoost(Attack):
    """Scale malicious updates by a constant factor to survive 1/N averaging.

    W_a = scale * W_honest. Small scale has no effect (1/N damping);
    large scale (>=10) dominates the aggregate and causes measurable damage.
    AUC=1.000 vs benign; c=50 -> +42% MSE damage (probe_diakrisis_damage).
    """

    scale: float = 10.0

    def craft(
        self,
        packages: OrderedDict[int, dict[str, Any]],
        malicious_ids: list[int],
        ctx: Any,
    ) -> OrderedDict[int, dict[str, Any]]:
        for cid in malicious_ids:
            pkg = packages[cid]
            pkg["regular_model_params"] = {
                k: v * self.scale for k, v in pkg["regular_model_params"].items()
            }
        return packages
