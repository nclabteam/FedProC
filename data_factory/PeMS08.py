import datetime
import os

import numpy as np
import polars as pl

from .base import BaseDataset


class PeMS08(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_raw = os.path.join("datasets", "PeMS08", "raw")
        self.path_temp = os.path.join("datasets", "PeMS08", "temp")
        self.save_path = os.path.join("datasets", "PeMS08")
        self.column_date = "Timestamp"
        self.column_target = ["Feature1", "Feature2", "Feature3"]
        self.column_train = ["Feature1", "Feature2", "Feature3"]
        self.granularity = 5
        self.granularity_unit = "minute"
        self.url = "https://raw.githubusercontent.com/guoshnBJTU/ASTGNN/refs/heads/main/data/PEMS08/PEMS08.npz"

    def download(self):
        # Create directories
        os.makedirs(self.path_raw, exist_ok=True)
        os.makedirs(self.path_temp, exist_ok=True)

        # Download the file
        file_path = os.path.join(self.path_temp, self.url.split("/")[-1])
        self.download_file(url=self.url, save_path=file_path)

        # Load the NumPy data
        data = np.load(file_path)["data"]  # Shape: (17856, 170, 3)

        # Generate timestamps
        start_time = datetime.datetime(2016, 7, 1, 0, 0, 0)
        time_delta = datetime.timedelta(minutes=self.granularity)
        dates = [start_time + i * time_delta for i in range(data.shape[0])]

        # Split the data into separate CSV files for each sensor
        for sensor_id in range(data.shape[1]):
            sensor_data = data[:, sensor_id, :]  # Extract data for the sensor

            # Create a Polars DataFrame
            df = pl.DataFrame(
                {
                    self.column_date: dates,
                    self.column_target[0]: sensor_data[:, 0],
                    self.column_target[1]: sensor_data[:, 1],
                    self.column_target[2]: sensor_data[:, 2],
                }
            )

            # Save the DataFrame to a CSV file
            df.write_csv(os.path.join(self.path_raw, f"{sensor_id}.csv"))
