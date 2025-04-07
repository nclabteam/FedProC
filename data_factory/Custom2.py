import os

from .base import CustomDataset
from .ETDataset import ETTh1, ETTh2


class Custom2(CustomDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = os.path.join("datasets", "Custom1")
        self.sets = [
            {
                "dataset": ETTh1,
                "output_len": 96,
            },
            {
                "dataset": ETTh2,
                "output_len": 192,
            },
        ]
