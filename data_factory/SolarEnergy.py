import gzip
import os
import shutil
from datetime import datetime, timedelta

import polars as pl
import requests

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
        df = pl.read_csv(extracted_path, separator=",", has_header=False)

        # Generate datetime column from 2015-01-01, with 1-hour increments
        start_date = datetime(2015, 1, 1, 0, 0)  # Start from 2015
        num_rows = df.shape[0]  # Number of rows in the dataset

        datetime_series = [start_date + timedelta(hours=i) for i in range(num_rows)]
        df = df.with_columns(pl.Series(self.column_date, datetime_series))

        # Save the dataset
        self.split_columns_into_files(
            df=df,
            path=self.path_raw,
            date_column=self.column_date,
            new_column_name="Value",
        )
