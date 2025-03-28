import os

import pandas as pd
import polars as pl

from .base import BaseDataset


class PeMSBAY(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_raw = os.path.join("datasets", "PeMSBAY", "raw")
        self.path_temp = os.path.join("datasets", "PeMSBAY", "temp")
        self.save_path = os.path.join("datasets", "PeMSBAY")
        self.column_date = "Datetime"
        self.column_target = ["Value"]
        self.column_train = ["Value"]
        self.granularity = 5
        self.granularity_unit = "minute"
        self.url = "1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq"

    def download(self):
        # Create directories
        os.makedirs(self.path_raw, exist_ok=True)
        os.makedirs(self.path_temp, exist_ok=True)

        # Download the dataset
        file_path = os.path.join(self.path_temp, "pems-bay.h5")
        self.download_from_google_drive(file_id=self.url, save_path=file_path)

        # Load the dataset
        df = pl.from_pandas(
            pd.read_hdf(file_path, key="speed"), include_index=True
        ).rename({"None": self.column_date})

        # Save the dataset
        self.split_columns_into_files(
            df=df,
            path=self.path_raw,
            date_column=self.column_date,
            new_column_name="Value",
            keep_old_column_name_as_filename=True,
        )
