import numpy as np

from .BaseScaler import BaseScaler, ScalerStats


class MinMax(BaseScaler):
    """Scale each feature to its observed minimum and maximum."""

    def __init__(self, stat: ScalerStats | None = None) -> None:
        super().__init__(stat=stat)
        if stat is not None:
            self.min, self.max = self.extract_statistics(
                stat=stat,
                names=("min", "max"),
            )

    def fit(self, data: np.ndarray) -> None:
        self.min = data.min(axis=0)
        self.max = data.max(axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * (self.max - self.min) + self.min
