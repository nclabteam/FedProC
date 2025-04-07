import os

from .base import CustomDataset
from .ETDataset import ETDatasetHour, ETDatasetMinute


class Custom1(CustomDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = os.path.join("datasets", "Custom1")
        self.sets = [
            {
                "dataset": ETDatasetHour,
                "column_target": ["OT"],
                "column_train": ["OT"],
            },
            {
                "dataset": ETDatasetMinute,
            },
        ]
