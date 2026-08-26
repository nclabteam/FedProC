from torch.nn import L1Loss


class MAE(L1Loss):
    """Compute mean absolute error."""
