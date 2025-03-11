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
                "seq_len": self.seq_len,
                "pred_len": self.pred_len,
                "offset_len": self.offset_len,
                "train_ratio": self.train_ratio,
            },
            {
                "dataset": ETDatasetMinute,
                "seq_len": self.seq_len,
                "pred_len": self.pred_len,
                "offset_len": self.offset_len,
                "train_ratio": self.train_ratio,
            },
        ]
