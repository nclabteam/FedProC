import os

import polars as pl

from .base import BaseDataset


class AppliancesEnergy(BaseDataset):
    """UCI household energy and environmental measurements."""

    measurement_columns = [
        "Appliances",
        "lights",
        "T1",
        "RH_1",
        "T2",
        "RH_2",
        "T3",
        "RH_3",
        "T4",
        "RH_4",
        "T5",
        "RH_5",
        "T6",
        "RH_6",
        "T7",
        "RH_7",
        "T8",
        "RH_8",
        "T9",
        "RH_9",
        "T_out",
        "Press_mm_hg",
        "RH_out",
        "Windspeed",
        "Visibility",
        "Tdewpoint",
    ]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_path = os.path.join("datasets", "AppliancesEnergy")
        self.path_raw = os.path.join(self.save_path, "raw")
        self.path_temp = os.path.join(self.save_path, "temp")

        self.column_date = "DateTime"
        self.column_target = list(self.measurement_columns)
        self.column_train = list(self.measurement_columns)
        self.granularity = 10
        self.granularity_unit = "minute"
        self.url = (
            "https://archive.ics.uci.edu/static/public/374/"
            "appliances+energy+prediction.zip"
        )

    def download(self) -> None:
        """Download UCI data and remove its synthetic noise variables."""
        os.makedirs(self.path_raw, exist_ok=True)
        os.makedirs(self.path_temp, exist_ok=True)
        self.download_and_extract(
            url=self.url,
            save_path=self.path_temp,
        )

        source_path = os.path.join(
            self.path_temp,
            "energydata_complete.csv",
        )
        frame = pl.read_csv(source_path).with_columns(
            pl.col("date").str.to_datetime("%Y-%m-%d %H:%M:%S").alias(self.column_date)
        )
        frame.select([self.column_date] + self.measurement_columns).write_csv(
            os.path.join(self.path_raw, "AppliancesEnergy.csv")
        )
