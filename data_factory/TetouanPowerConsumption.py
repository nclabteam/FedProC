import os

import polars as pl

from .base import BaseDataset


class TetouanPowerConsumption(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_raw = os.path.join("datasets", "TetouanPowerConsumption", "raw")
        self.path_temp = os.path.join("datasets", "TetouanPowerConsumption", "temp")
        self.save_path = os.path.join("datasets", "TetouanPowerConsumption")
        self.column_date = "DateTime"
        self.column_target = [
            "Temperature",
            "Humidity",
            "Wind Speed",
            "general diffuse flows",
            "diffuse flows",
            "Power Consumption",
        ]
        self.column_train = [
            "Temperature",
            "Humidity",
            "Wind Speed",
            "general diffuse flows",
            "diffuse flows",
            "Power Consumption",
        ]
        self.granularity = 10
        self.granularity_unit = "minute"
        self.url = "https://archive.ics.uci.edu/static/public/849/power+consumption+of+tetouan+city.zip"

    def download(self):
        # Create directories
        os.makedirs(self.path_temp, exist_ok=True)
        os.makedirs(self.path_raw, exist_ok=True)

        # Download and extract the data
        self.download_and_extract(url=self.url, save_path=self.path_temp)

        # Load the data
        df = pl.read_csv(
            os.path.join(self.path_temp, "Tetuan City power consumption.csv")
        )

        # Rename the columns
        df = df.with_columns(pl.col("DateTime").str.to_datetime("%m/%d/%Y %H:%M"))

        # Save the data
        zones = [
            "Zone 1 Power Consumption",
            "Zone 2  Power Consumption",
            "Zone 3  Power Consumption",
        ]
        for idx, zone in enumerate(zones):
            sdf = df.rename({zone: "Power Consumption"})
            for szone in zones:
                if szone != zone:
                    sdf = sdf.drop(szone)
            sdf.write_csv(os.path.join(self.path_raw, f"{idx+1}.csv"))
