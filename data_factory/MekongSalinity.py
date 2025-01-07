import os

from .base import BaseDataset


class MekongSalinity(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_raw = os.path.join("datasets", "MekongSalinity", "raw")
        self.save_path = os.path.join("datasets", "MekongSalinity")
        self.column_date = "date"
        self.column_target = ["average"]
        self.column_train = ["average"]
        self.granularity = 1
        self.granularity_unit = "day"
