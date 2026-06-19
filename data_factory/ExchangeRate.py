import os
from datetime import datetime, timedelta

import polars as pl

from .base import BaseDataset


class ExchangeRate(BaseDataset):
    """
    Daily exchange rates for 8 foreign countries (1990-2016).

    Countries: Australia, British, Canada, Switzerland, China, Japan, New Zealand, Singapore.
    Data is preprocessed with weekends/holidays already removed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = os.path.join("datasets", "ExchangeRate")
        self.path_raw = os.path.join("datasets", "ExchangeRate", "raw")
        self.path_temp = os.path.join("datasets", "ExchangeRate", "temp")
        self.column_date = "Date"
        self.column_target = ["Value"]
        self.column_train = ["Value"]
        self.granularity = 1
        self.granularity_unit = "day"
        self.url = "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/refs/heads/master/exchange_rate/exchange_rate.txt.gz"
        self.split_files = True

    def download(self):
        os.makedirs(self.path_raw, exist_ok=True)
        extracted_path = self.download_and_extract_gz(url=self.url, save_dir=self.path_temp)
        df = pl.scan_csv(extracted_path, separator=",", has_header=False).collect()
        start_date = datetime(1990, 1, 1)
        df = df.with_columns(
            pl.Series(
                self.column_date,
                [start_date + timedelta(days=i) for i in range(df.shape[0])],
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
            df.write_csv(os.path.join(self.path_raw, "exchange_rate.csv"))


class ExchangeRateOG(ExchangeRate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = os.path.join("datasets", "ExchangeRateOG")
        self.path_raw = os.path.join("datasets", "ExchangeRateOG", "raw")
        self.path_temp = os.path.join("datasets", "ExchangeRateOG", "temp")
        self.column_target = [f"V{i:03d}" for i in range(1, 9)]
        self.column_train = [f"V{i:03d}" for i in range(1, 9)]
        self.split_files = False
