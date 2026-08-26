import os

from .base import CustomDataset
from .Electricity import Electricity
from .ETDataset import ETDatasetHour
from .SolarEnergy import SolarEnergy
from .TetouanPowerConsumption import TetouanPowerConsumption


class Customized1(CustomDataset):
    """Combine four heterogeneous energy forecasting datasets."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_path = os.path.join("datasets", "Customized1")
        self.sets = [
            {
                "dataset": ETDatasetHour,
            },
            {
                "dataset": TetouanPowerConsumption,
            },
            {
                "dataset": SolarEnergy,
            },
            {
                "dataset": Electricity,
            },
        ]


class Customized2(CustomDataset):
    """Combine univariate and multivariate ETDataset clients."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_path = os.path.join("datasets", "Customized2")
        self.sets = [
            {
                "dataset": ETDatasetHour,
                "column_target": ["OT"],
                "column_train": ["OT"],
            },
            {
                "dataset": ETDatasetHour,
            },
        ]


class Customized3(CustomDataset):
    """Combine datasets configured with different forecast horizons."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_path = os.path.join("datasets", "Customized3")
        self.sets = [
            {
                "dataset": ETDatasetHour,
                "output_len": 96,
            },
            {
                "dataset": TetouanPowerConsumption,
                "output_len": 192,
            },
        ]
