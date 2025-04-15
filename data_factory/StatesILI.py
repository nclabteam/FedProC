import os

import polars as pl

from .base import BaseDataset


class StatesILI(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set the dataset name and path
        self.save_path = os.path.join("datasets", "StatesILI")
        self.path_raw = os.path.join("datasets", "StatesILI", "raw")
        self.path_temp = os.path.join("datasets", "StatesILI", "temp")

        # Set the dataset parameters
        self.column_date = "date"
        self.column_target = ["Value"]
        self.column_train = ["Value"]
        self.granularity = 1
        self.granularity_unit = "week"
        self.url = "https://raw.githubusercontent.com/emilylaiken/ml-flu-prediction/refs/heads/master/data/States_ILI.csv"

    def download(self):
        # Create the directory if it doesn't exist
        os.makedirs(self.path_raw, exist_ok=True)
        os.makedirs(self.path_temp, exist_ok=True)

        # Download the file
        file_name = self.url.split("/")[-1]
        file_path = os.path.join(self.path_temp, file_name)
        self.download_file(url=self.url, save_path=file_path)

        # Read the CSV file
        df = pl.read_csv(file_path)

        # Convert the date column (MM/DD/YY) to datetime format
        df = df.with_columns(pl.col(self.column_date).str.to_datetime("%m/%d/%y"))

        # Save the dataset as multiple files
        self.split_columns_into_files(
            df=df,
            path=self.path_raw,
            date_column=self.column_date,
            new_column_name="Value",
            keep_old_column_name_as_filename=True,
        )
