import os

from .base import CustomDataset
from .Electricity import Electricity
from .ETDataset import ETDatasetHour
from .SolarEnergy import SolarEnergy
from .TetouanPowerConsumption import TetouanPowerConsumption


class Customized1(CustomDataset):
    def __init__(self, *args, **kwargs):
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
    def __init__(self, *args, **kwargs):
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
    def __init__(self, *args, **kwargs):
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
