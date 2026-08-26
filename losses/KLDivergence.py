from typing import Any

from torch.nn import KLDivLoss


class KLDivergence(KLDivLoss):
    """Compute KL divergence using its mathematical batch-mean reduction."""

    generic_eval = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(reduction="batchmean", **kwargs)
