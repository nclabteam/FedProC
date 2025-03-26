import datetime
import os

import numpy as np
import polars as pl

from .base import BaseDataset


class PEMS03(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_raw = os.path.join("datasets", "PEMS03", "raw")
        self.path_temp = os.path.join("datasets", "PEMS03", "temp")
        self.save_path = os.path.join("datasets", "PEMS03")
        self.column_date = "Timestamp"
        self.column_target = ["Value"]
        self.column_train = ["Value"]
        self.granularity = 5
        self.granularity_unit = "minute"
        self.url = "https://raw.githubusercontent.com/guoshnBJTU/ASTGNN/refs/heads/main/data/PEMS03/PEMS03.npz"
        self.metadata = "https://raw.githubusercontent.com/guoshnBJTU/ASTGNN/refs/heads/main/data/PEMS03/PEMS03.txt"

    def download(self):
        os.makedirs(self.path_raw, exist_ok=True)
        os.makedirs(self.path_temp, exist_ok=True)

        file_path = os.path.join(self.path_temp, self.url.split("/")[-1])
        self.download_file(url=self.url, save_path=file_path)

        meta_path = os.path.join(self.save_path, self.metadata.split("/")[-1])
        self.download_file(url=self.metadata, save_path=meta_path)

        # Load the NumPy data
        data = np.load(file_path)["data"]  # Shape: (26208, 358, 1)
        data = data.squeeze(axis=-1)  # Convert from (26208, 358, 1) to (26208, 358)

        # Load sensor IDs from metadata
        sensor_ids = (
            pl.read_csv(meta_path, has_header=False, new_columns=["Sensor_ID"])
            .to_series()
            .cast(pl.Utf8)
            .to_list()
        )

        # Generate timestamps
        start_time = datetime.datetime(2018, 9, 1, 0, 0, 0)
        time_index = [
            start_time + datetime.timedelta(minutes=5 * i) for i in range(data.shape[0])
        ]

        # Convert to Polars DataFrame with correct schema
        df = pl.DataFrame(data, schema=sensor_ids)

        # Add timestamp column
        df = df.with_columns(pl.Series(self.column_date, time_index))

        # Move Timestamp to the first column
        df = df.select([self.column_date] + df.columns[:-1])

        # Save the dataset
        self.split_columns_into_files(
            df=df,
            path=self.path_raw,
            date_column=self.column_date,
            new_column_name="Value",
            keep_old_column_name_as_filename=True,
        )
