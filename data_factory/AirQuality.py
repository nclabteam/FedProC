import os

import polars as pl

from .base import BaseDataset


class AirQuality(BaseDataset):
    """UCI Air Quality hourly sensor and pollutant measurements."""

    measurement_columns = [
        "CO(GT)",
        "PT08.S1(CO)",
        "C6H6(GT)",
        "PT08.S2(NMHC)",
        "NOx(GT)",
        "PT08.S3(NOx)",
        "NO2(GT)",
        "PT08.S4(NO2)",
        "PT08.S5(O3)",
        "T",
        "RH",
        "AH",
    ]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_path = os.path.join("datasets", "AirQuality")
        self.path_raw = os.path.join(self.save_path, "raw")
        self.path_temp = os.path.join(self.save_path, "temp")

        self.column_date = "DateTime"
        self.column_target = list(self.measurement_columns)
        self.column_train = list(self.measurement_columns)
        self.granularity = 1
        self.granularity_unit = "hour"
        self.url = "https://archive.ics.uci.edu/static/public/360/" "air+quality.zip"

    def download(self) -> None:
        """Download UCI data and normalize its locale-specific CSV."""
        os.makedirs(self.path_raw, exist_ok=True)
        os.makedirs(self.path_temp, exist_ok=True)
        self.download_and_extract(
            url=self.url,
            save_path=self.path_temp,
        )

        source_path = os.path.join(
            self.path_temp,
            "AirQualityUCI.csv",
        )
        frame = pl.read_csv(
            source_path,
            separator=";",
            decimal_comma=True,
            null_values=["-200", "-200.0"],
            truncate_ragged_lines=True,
        )
        frame = frame.filter(pl.col("Date").is_not_null())
        frame = frame.with_columns(
            pl.concat_str(
                [
                    pl.col("Date"),
                    pl.lit(" "),
                    pl.col("Time"),
                ]
            )
            .str.to_datetime("%d/%m/%Y %H.%M.%S")
            .alias(self.column_date)
        )
        frame.select([self.column_date] + self.measurement_columns).write_csv(
            os.path.join(self.path_raw, "AirQuality.csv")
        )
