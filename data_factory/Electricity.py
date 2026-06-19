import os
from datetime import datetime, timedelta

import polars as pl

from .base import BaseDataset


class Electricity(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = os.path.join("datasets", "Electricity")
        self.path_raw = os.path.join("datasets", "Electricity", "raw")
        self.path_temp = os.path.join("datasets", "Electricity", "temp")
        self.column_date = "Date"
        self.column_target = ["Value"]
        self.column_train = ["Value"]
        self.granularity = 15
        self.granularity_unit = "minute"
        self.url = "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/refs/heads/master/electricity/electricity.txt.gz"
        self.split_files = True

    def download(self):
        os.makedirs(self.path_raw, exist_ok=True)
        extracted_path = self.download_and_extract_gz(url=self.url, save_dir=self.path_temp)
        df = pl.scan_csv(extracted_path, separator=",", has_header=False).collect()
        start_date = datetime(2012, 1, 1, 0, 15)
        df = df.with_columns(
            pl.Series(
                self.column_date,
                [
                    start_date + timedelta(minutes=i * self.granularity)
                    for i in range(df.shape[0])
                ],
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
            df.write_csv(os.path.join(self.path_raw, "electricity.csv"))
