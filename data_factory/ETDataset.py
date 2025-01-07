import os

from .base import BaseDataset


class ETDatasetMinute(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_raw = os.path.join("datasets", "ETDataset", "raw", "minute")
        self.save_path = os.path.join("datasets", "ETDataset", "ETDatasetMinute")
        self.column_date = "date"
        self.column_target = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
        self.column_train = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
        self.granularity = 15
        self.granularity_unit = "minute"
        self.urls = [
            "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/refs/heads/main/ETT-small/ETTm1.csv",
            "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/refs/heads/main/ETT-small/ETTm2.csv",
        ]

    def download(self):
        # Create the directory if it doesn't exist
        os.makedirs(self.path_raw, exist_ok=True)

        for url in self.urls:
            # Extract the file name from the URL
            file_name = url.split("/")[-1]
            file_path = os.path.join(self.path_raw, file_name)
            self.download_file(url=url, save_path=file_path)


class ETDatasetHour(ETDatasetMinute, BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_raw = os.path.join("datasets", "ETDataset", "raw", "hour")
        self.save_path = os.path.join("datasets", "ETDataset", "ETDatasetHour")
        self.column_date = "date"
        self.column_target = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
        self.column_train = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
        self.granularity = 1
        self.granularity_unit = "hour"
        self.urls = [
            "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/refs/heads/main/ETT-small/ETTh1.csv",
            "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/refs/heads/main/ETT-small/ETTh2.csv",
        ]
