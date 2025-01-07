import os

from .base import BaseDataset


class Weather5K(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_raw = os.path.join("datasets", "Weather5K", "raw")
        self.save_path = os.path.join("datasets", "Weather5K")
        self.column_date = "DATE"
        self.column_target = ["TMP", "DEW", "WND_ANGLE", "WND_RATE", "SLP"]
        self.column_train = ["TMP", "DEW", "WND_ANGLE", "WND_RATE", "SLP"]
        self.granularity = 1
        self.granularity_unit = "hour"
