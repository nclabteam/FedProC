import numpy as np

from .BaseScaler import BaseScaler, ScalerStats


class MaxAbs(BaseScaler):
    """Scale each feature by its maximum absolute value."""

    def __init__(self, stat: ScalerStats | None = None) -> None:
        super().__init__(stat=stat)
        if stat is not None:
            (self.max_abs,) = self.extract_statistics(
                stat=stat,
                names=("max_abs",),
            )

    def fit(self, data: np.ndarray) -> None:
        self.max_abs = np.max(np.abs(data), axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return self.divide_no_nan(a=data, b=self.max_abs)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.max_abs
