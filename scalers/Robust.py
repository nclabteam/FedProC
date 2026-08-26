import numpy as np

from .BaseScaler import BaseScaler, ScalerStats


class Robust(BaseScaler):
    """Scale each feature by its interquartile range."""

    def __init__(self, stat: ScalerStats | None = None) -> None:
        super().__init__(stat=stat)
        if stat is not None:
            self.q1s, self.q3s = self.extract_statistics(
                stat=stat,
                names=("q1", "q3"),
            )

    def fit(self, data: np.ndarray) -> None:
        self.q1s = np.percentile(data, 25, axis=0)
        self.q3s = np.percentile(data, 75, axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.q1s) / (self.q3s - self.q1s)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * (self.q3s - self.q1s) + self.q1s
