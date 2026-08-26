import numpy as np

from .BaseScaler import BaseScaler, ScalerStats


class Standard(BaseScaler):
    """Scale each feature to zero mean and unit variance."""

    def __init__(self, stat: ScalerStats | None = None) -> None:
        super().__init__(stat=stat)
        if stat is not None:
            self.mean, self.std = self.extract_statistics(
                stat=stat,
                names=("mean", "std"),
            )

    def fit(self, data: np.ndarray) -> None:
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        return self.divide_no_nan(
            a=np.asarray(data, dtype=np.float32) - mean,
            b=std,
        )

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return (
            np.asarray(data, dtype=np.float32) * np.asarray(self.std, dtype=np.float32)
        ) + np.asarray(self.mean, dtype=np.float32)
