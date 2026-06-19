import os
from datetime import datetime, timedelta

import polars as pl

from .base import BaseDataset


class SolarEnergy(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = os.path.join("datasets", "SolarEnergy")
        self.path_raw = os.path.join("datasets", "SolarEnergy", "raw")
        self.path_temp = os.path.join("datasets", "SolarEnergy", "temp")
        self.column_date = "Date"
        self.column_target = ["Value"]
        self.column_train = ["Value"]
        self.granularity = 1
        self.granularity_unit = "hour"
        self.url = "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/solar-energy/solar_AL.txt.gz"
        self.split_files = True

    def download(self):
        os.makedirs(self.path_raw, exist_ok=True)
        extracted_path = self.download_and_extract_gz(url=self.url, save_dir=self.path_temp)
        df = pl.read_csv(extracted_path, separator=",", has_header=False)
        start_date = datetime(2015, 1, 1, 0, 0)
        df = df.with_columns(
            pl.Series(
                self.column_date,
                [start_date + timedelta(hours=i) for i in range(df.shape[0])],
            )
        )
        if self.split_files:
            self.split_columns_into_files(
                df=df,
                path=self.path_raw,
                date_column=self.column_date,
                new_column_name="Value",
            )
        else:
            df.write_csv(os.path.join(self.path_raw, "solar_energy.csv"))


class SolarEnergyOG(SolarEnergy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = os.path.join("datasets", "SolarEnergyOG")
        self.path_raw = os.path.join("datasets", "SolarEnergyOG", "raw")
        self.path_temp = os.path.join("datasets", "SolarEnergyOG", "temp")
        self.column_target = [f"column_{i}" for i in range(1, 138)]
        self.column_train = [f"column_{i}" for i in range(1, 138)]
        self.split_files = False
