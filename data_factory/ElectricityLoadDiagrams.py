import os

import pandas as pd
import polars as pl

from .base import BaseDataset


class ElectricityLoadDiagrams(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_raw = os.path.join("datasets", "ElectricityLoadDiagrams", "raw")
        self.path_temp = os.path.join("datasets", "ElectricityLoadDiagrams", "temp")
        self.save_path = os.path.join("datasets", "ElectricityLoadDiagrams")
        self.column_date = "Date"
        self.column_target = ["Value"]
        self.column_train = ["Value"]
        self.granularity = 15
        self.granularity_unit = "minute"
        self.url = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"

    def download(self):
        os.makedirs(self.path_temp, exist_ok=True)
        os.makedirs(self.path_raw, exist_ok=True)

        self.download_and_extract(url=self.url, save_path=self.path_temp)

        df = pd.read_csv(
            os.path.join(self.path_temp, "LD2011_2014.txt"), sep=";", decimal=","
        )
        df = df.rename(columns={df.columns[0]: self.column_date})
        df = pl.DataFrame(df).with_columns(
            pl.col(self.column_date).str.to_datetime("%Y-%m-%d %H:%M:%S")
        )

        # Save the data
        self.split_columns_into_files(
            df=df,
            path=self.path_raw,
            date_column=self.column_date,
            new_column_name="Value",
        )
