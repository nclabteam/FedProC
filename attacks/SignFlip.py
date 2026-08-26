from collections import OrderedDict
from typing import Any

from .base import Attack


class SignFlip(Attack):
    """Flip the sign of all regular model parameters for malicious clients.

    Each malicious client sends -W instead of W. Cheap to compute;
    destroys the gradient signal. AUC=1.000 vs benign (probe_diakrisis_kill).
    """

    def craft(
        self,
        packages: OrderedDict[int, dict[str, Any]],
        malicious_ids: list[int],
        ctx: Any,
    ) -> OrderedDict[int, dict[str, Any]]:
        for cid in malicious_ids:
            pkg = packages[cid]
            pkg["regular_model_params"] = {
                k: -v for k, v in pkg["regular_model_params"].items()
            }
        return packages
