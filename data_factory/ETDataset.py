import os

from .base import BaseDataset, CustomDataset


class ETDatasetMinute(CustomDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = os.path.join("datasets", "ETDataset", "ETDatasetMinute")
        self.sets = [
            {
                "dataset": ETTm1,
            },
            {
                "dataset": ETTm2,
            },
        ]


class ETDatasetHour(ETDatasetMinute, BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = os.path.join("datasets", "ETDataset", "ETDatasetHour")
        self.sets = [
            {
                "dataset": ETTh1,
            },
            {
                "dataset": ETTh2,
            },
        ]


class ETTm1(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_raw = os.path.join("datasets", "ETDataset", "raw", "ETTm1")
        self.save_path = os.path.join("datasets", "ETDataset", "ETTm1")
        self.column_date = "date"
        self.column_target = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
        self.column_train = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
        self.granularity = 15
        self.granularity_unit = "minute"
        self.url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/refs/heads/main/ETT-small/ETTm1.csv"

    def download(self):
        os.makedirs(self.path_raw, exist_ok=True)
        file_name = self.url.split("/")[-1]
        file_path = os.path.join(self.path_raw, file_name)
        self.download_file(url=self.url, save_path=file_path)


class ETTm2(ETTm1, BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_raw = os.path.join("datasets", "ETDataset", "raw", "ETTm2")
        self.save_path = os.path.join("datasets", "ETDataset", "ETTm2")
        self.column_date = "date"
        self.column_target = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
        self.column_train = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
        self.granularity = 15
        self.granularity_unit = "minute"
        self.url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/refs/heads/main/ETT-small/ETTm2.csv"


class ETTh1(ETTm1, BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_raw = os.path.join("datasets", "ETDataset", "raw", "ETTh1")
        self.save_path = os.path.join("datasets", "ETDataset", "ETTh1")
        self.column_date = "date"
        self.column_target = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
        self.column_train = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
        self.granularity = 1
        self.granularity_unit = "hour"
        self.url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/refs/heads/main/ETT-small/ETTh1.csv"


class ETTh2(ETTm1, BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_raw = os.path.join("datasets", "ETDataset", "raw", "ETTh2")
        self.save_path = os.path.join("datasets", "ETDataset", "ETTh2")
        self.column_date = "date"
        self.column_target = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
        self.column_train = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
        self.granularity = 1
        self.granularity_unit = "hour"
        self.url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/refs/heads/main/ETT-small/ETTh2.csv"
