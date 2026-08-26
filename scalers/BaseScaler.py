from collections.abc import Mapping

import numpy as np

ScalerStats = Mapping[str, Mapping[str, float]]


class BaseScaler:
    """Identity scaler with shared statistic helpers."""

    def __init__(self, stat: ScalerStats | None = None) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> None:
        pass

    def transform(self, data: np.ndarray) -> np.ndarray:
        return data

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data

    @staticmethod
    def extract_statistics(
        stat: ScalerStats,
        names: tuple[str, ...],
    ) -> tuple[np.ndarray, ...]:
        """Extract multiple per-feature statistics in one pass."""
        columns: list[list[float]] = [[] for _ in names]
        for feature in stat.values():
            for column, name in zip(columns, names):
                column.append(feature[name])
        return tuple(np.asarray(column) for column in columns)

    @staticmethod
    def divide_no_nan(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Divide arrays and replace non-finite results with zero."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.nan_to_num(a / b, nan=0.0, posinf=0.0, neginf=0.0)
