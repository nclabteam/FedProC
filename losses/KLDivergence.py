from torch.nn import KLDivLoss


class KLDivergence(KLDivLoss):
    def __init__(self, **kwargs):
        # Use 'batchmean' reduction to avoid the deprecation warning
        # and to align with the mathematical definition of KL divergence
        super().__init__(reduction="batchmean", **kwargs)
