import os

from .base import BaseDataset


class BaseStation5G(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = os.path.join("datasets", "BaseStation5G")
        self.path_raw = os.path.join("datasets", "BaseStation5G", "raw")
        self.path_temp = os.path.join("datasets", "BaseStation5G", "temp")
        self.column_date = "time"
        self.column_target = [
            "down",
            "up",
            "rnti_count",
            "mcs_down",
            "mcs_down_var",
            "mcs_up",
            "mcs_up_var",
            "rb_down",
            "rb_down_var",
            "rb_up",
            "rb_up_var",
        ]
        self.column_train = [
            "down",
            "up",
            "rnti_count",
            "mcs_down",
            "mcs_down_var",
            "mcs_up",
            "mcs_up_var",
            "rb_down",
            "rb_down_var",
            "rb_up",
            "rb_up_var",
        ]
        self.granularity = 2
        self.granularity_unit = "minute"
        self.url = "https://raw.githubusercontent.com/vperifan/Federated-Time-Series-Forecasting/refs/heads/main/dataset/full_dataset.csv"

    def download(self):
        # Create the directories
        os.makedirs(self.path_raw, exist_ok=True)
        os.makedirs(self.path_temp, exist_ok=True)

        # Download the file
        file_name = self.url.split("/")[-1]
        file_path = os.path.join(self.path_temp, file_name)
        self.download_file(url=self.url, save_path=file_path)

        # split into stations
        df = self.read(file_path)
        for station in df["District"].unique().to_list():
            sdf = df.filter(df["District"] == station)
            sdf = sdf.drop("District")
            sdf.write_csv(os.path.join(self.path_raw, f"{station}.csv"))
