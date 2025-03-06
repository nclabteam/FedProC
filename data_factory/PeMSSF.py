import os
from datetime import datetime, timedelta

import holidays
import numpy as np
import polars as pl

from .base import BaseDataset


class PeMSSF(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_raw = os.path.join("datasets", "PeMSSF", "raw")
        self.path_temp = os.path.join("datasets", "PeMSSF", "temp")
        self.save_path = os.path.join("datasets", "PeMSSF")
        self.column_date = "datetime"
        self.column_target = ["occupancy_rate"]
        self.column_train = ["occupancy_rate"]
        self.granularity = 10
        self.granularity_unit = "minute"
        self.url = "https://archive.ics.uci.edu/static/public/204/pems+sf.zip"

    @staticmethod
    def create_dataframe(file_path, stations_list_file_path):
        def parse_matrix(matrix_str):
            # Remove leading/trailing brackets and split by semicolon to separate rows
            rows = matrix_str.strip()[1:-1].split(";")

            # Split rows by space to get individual values and convert to float
            matrix = np.array([[float(value) for value in row.split()] for row in rows])

            return matrix

        # Read data file
        with open(file_path, "r") as file:
            data = [parse_matrix(line) for line in file]

        data = np.array(data)

        # Reshape the array to make it compatible with DataFrame creation
        num_records, num_stations, num_timestamps = data.shape
        reshaped_data = data.reshape(num_records, -1)

        # Read the stations_list file to get station numbers
        with open(stations_list_file_path, "r") as stations_file:
            station_numbers = (
                stations_file.read().replace("[", "").replace("]", "").split()
            )

        # Convert station numbers to integers
        stations_list = [int(station) for station in station_numbers]

        # Create DataFrame from the reshaped data with station numbers as column names
        column_names = [
            f"station_{station}_timestamp_{j}"
            for station in stations_list
            for j in range(num_timestamps)
        ]
        data = pl.DataFrame(reshaped_data, schema=column_names)

        # Check the shape of the created DataFrame
        print("Shape of DataFrame:", data.shape)

        return data

    def download(self):
        # Create the directories
        os.makedirs(self.path_temp, exist_ok=True)
        os.makedirs(self.path_raw, exist_ok=True)

        # Download the file
        self.download_and_extract(url=self.url, save_path=self.path_temp)

        # Read the files
        df_train = self.create_dataframe(
            file_path=os.path.join(self.path_temp, "PEMS_train"),
            stations_list_file_path=os.path.join(self.path_temp, "stations_list"),
        )
        df_train = df_train.with_columns(
            pl.Series("day", range(df_train.height)).cast(pl.Int32)
        )
        df_test = self.create_dataframe(
            file_path=os.path.join(self.path_temp, "PEMS_test"),
            stations_list_file_path=os.path.join(self.path_temp, "stations_list"),
        )
        df_test = df_test.with_columns(
            pl.Series(
                "day", range(df_train.height, df_train.height + df_test.height)
            ).cast(pl.Int32)
        )
        df = pl.concat([df_train, df_test], how="vertical")
        del df_train, df_test

        # Convert wide format to long format
        df = df.unpivot(
            index=["day"],
            variable_name="station_timestamp",
            value_name="occupancy_rate",
        )

        # Extract station ID and timestamp using string operations
        df = df.with_columns(
            [
                df["station_timestamp"]
                .str.extract(r"station_(\d+)_timestamp_\d+")
                .alias("station")
                .cast(pl.Int32),
                df["station_timestamp"]
                .str.extract(r"timestamp_(\d+)")
                .alias("timestamp")
                .cast(pl.Int32),
            ]
        )

        # Convert day and timestamp to datetime
        df = df.with_columns(
            pl.struct(["day", "timestamp"])
            .map_elements(
                lambda s: datetime(year=2008, month=1, day=1)
                + timedelta(days=s["day"])
                + timedelta(minutes=s["timestamp"] * self.granularity),
                return_dtype=pl.Datetime,
            )
            .alias(self.column_date)
        )

        # Reorder the columns and drop the unnecessary columns
        df = df.select(
            [
                self.column_date,
                "station",
                "occupancy_rate",
            ]
        )

        # Get the holidays
        huss = [
            datetime(year=ptr[0].year, month=ptr[0].month, day=ptr[0].day)
            for ptr in holidays.US(years=[2008, 2009]).items()
        ] + [datetime(year=2008, month=3, day=9), datetime(year=2009, month=3, day=8)]
        huss = sorted(huss)

        # Remove rows corresponding to holidays
        for holiday in huss:
            # Filter rows that come after the holiday
            df_shifted = df.filter(pl.col(self.column_date) > holiday)

            # Shift those rows to the next day by converting the datetime to timestamp, adding 1 day in microseconds, and converting it back
            df_shifted = df_shifted.with_columns(
                (
                    pl.col(self.column_date).cast(pl.Int64) + (24 * 60 * 60 * 1_000_000)
                ).cast(pl.Datetime)
            )

            # Remove the rows for the holiday (since they are moved to the next day)
            df = df.filter(pl.col(self.column_date) < holiday).vstack(df_shifted)

        # Split into stations
        self.split_column_into_files(
            df=df,
            path=self.path_raw,
            station_column="station",
            date_column=self.column_date,
            remove_station_column=True,
        )
