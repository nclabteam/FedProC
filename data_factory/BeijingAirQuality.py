import io
import os
import zipfile

import polars as pl
import requests

from .base import BaseDataset


class BejingAirQuality(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_raw = os.path.join("datasets", "BejingAirQuality", "raw")
        self.save_path = os.path.join("datasets", "BejingAirQuality")
        self.column_date = "Date"
        self.column_target = [
            "PM2.5",
            "PM10",
            "SO2",
            "NO2",
            "CO",
            "O3",
            "TEMP",
            "PRES",
            "DEWP",
            "RAIN",
            "WSPM",
        ]
        self.column_train = [
            "PM2.5",
            "PM10",
            "SO2",
            "NO2",
            "CO",
            "O3",
            "TEMP",
            "PRES",
            "DEWP",
            "RAIN",
            "WSPM",
        ]
        self.granularity = 1
        self.granularity_unit = "hour"
        self.url = "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"

    def download(self):
        tpath = os.path.join(self.save_path, "temp")
        os.makedirs(tpath, exist_ok=True)

        response = requests.get(self.url)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(tpath)

        with zipfile.ZipFile(
            os.path.join(tpath, "PRSA2017_Data_20130301-20170228.zip")
        ) as zip_ref:
            zip_ref.extractall(self.save_path)
        os.rename(
            src=os.path.join(self.save_path, "PRSA_Data_20130301-20170228"),
            dst=self.path_raw,
        )

    def read(self, path):
        try:
            df = pl.read_csv(
                path,
                try_parse_dates=True,
                null_values=["NA"],
                dtypes={
                    "SO2": pl.Float64,
                    "NO2": pl.Float64,
                    "RAIN": pl.Float64,
                    "O3": pl.Float64,
                    "PM10": pl.Float64,
                    "PM2.5": pl.Float64,
                },
            )
            return df
        except pl.exceptions.NoDataError:
            print(f"Empty file: {path}")
            return None

    def prepossess(self, df):
        df = super().prepossess(df)
        # merge  year ┆ month ┆ day ┆ hour into self.column_date
        df = df.with_columns(
            pl.concat_str(
                [
                    pl.col("year").cast(pl.Utf8),
                    pl.col("month").cast(pl.Utf8).str.zfill(2),
                    pl.col("day").cast(pl.Utf8).str.zfill(2),
                    pl.col("hour").cast(pl.Utf8).str.zfill(2),
                    pl.lit("0000").cast(pl.Utf8),
                ]
            )
            .alias(self.column_date)
            .str.to_datetime("%Y%m%d%H%M%S")  # Convert to datetime
        )
        return df
