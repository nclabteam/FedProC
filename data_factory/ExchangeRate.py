import gzip
import os
import shutil
from datetime import datetime, timedelta

import polars as pl
import requests

from .base import BaseDataset


class ExchangeRate(BaseDataset):
    """
    Daily exchange rates for 8 foreign countries (1990-2016).
    
    Countries: Australia, British, Canada, Switzerland, China, Japan, New Zealand, Singapore.
    Data is preprocessed with weekends/holidays already removed.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set the paths for the dataset
        self.save_path = os.path.join("datasets", "ExchangeRate")
        self.path_raw = os.path.join("datasets", "ExchangeRate", "raw")
        self.path_temp = os.path.join("datasets", "ExchangeRate", "temp")

        # Set the dataset parameters
        self.column_date = "Date"
        self.column_target = ["Value"]
        self.column_train = ["Value"]
        self.granularity = 1
        self.granularity_unit = "day"
        self.url = "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/refs/heads/master/exchange_rate/exchange_rate.txt.gz"
        self.split_files = True

    def download(self):
        # Create the directory if it doesn't exist
        os.makedirs(self.path_raw, exist_ok=True)
        os.makedirs(self.path_temp, exist_ok=True)

        # Define the output file name
        output_file = os.path.basename(self.url)
        temp_path = os.path.join(self.path_temp, output_file)
        extracted_path = os.path.join(self.path_temp, output_file.replace(".gz", ""))

        # Download the file
        response = requests.get(self.url, stream=True)
        if response.status_code == 200:
            with open(temp_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
        else:
            raise Exception(f"Failed to download file from {self.url}")

        # Extract the file
        with gzip.open(temp_path, "rb") as f_in:
            with open(extracted_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Load the dataset
        df = pl.scan_csv(extracted_path, separator=",", has_header=False).collect()

        # Generate datetime column from 1990-01-01, with daily increments
        start_date = datetime(1990, 1, 1)
        num_rows = df.shape[0]  # Number of rows in the dataset

        datetime_series = [
            start_date + timedelta(days=i)
            for i in range(num_rows)
        ]
        df = df.with_columns(pl.Series(self.column_date, datetime_series))

        if self.split_files:
            # Save the dataset as multiple files
            self.split_columns_into_files(
                df=df,
                path=self.path_raw,
                date_column=self.column_date,
                new_column_name="Value",
            )
        else:
            # Save the dataset as a single file
            df.write_csv(os.path.join(self.path_raw, "exchange_rate.csv"))


class ExchangeRateOG(ExchangeRate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set the paths for the dataset
        self.save_path = os.path.join("datasets", "ExchangeRateOG")
        self.path_raw = os.path.join("datasets", "ExchangeRateOG", "raw")
        self.path_temp = os.path.join("datasets", "ExchangeRateOG", "temp")

        # Set the dataset parameters
        self.column_target = [f"V{i:03d}" for i in range(1, 9)]
        self.column_train = [f"V{i:03d}" for i in range(1, 9)]
        self.split_files = False