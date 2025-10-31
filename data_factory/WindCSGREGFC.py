import os

import polars as pl

from .base import BaseDataset


class WindCSGREGFC(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set the dataset name and path
        self.save_path = os.path.join("datasets", "WindCSGREGFC")
        self.path_raw = os.path.join("datasets", "WindCSGREGFC", "raw")
        self.path_temp = os.path.join("datasets", "WindCSGREGFC", "temp")

        # Set the dataset parameters
        self.column_date = "Time(year-month-day h:m:s)"
        self.column_target = [
            "Wind speed at height of 10 meters (m/s)",
            "Wind direction at height of 10 meters (˚)",
            "Wind speed at height of 30 meters (m/s)",
            "Wind direction at height of 30 meters (˚)",
            "Wind speed at height of 50 meters (m/s)",
            "Wind direction at height of 50 meters (˚)",
            "Wind speed - at the height of wheel hub (m/s)",
            "Wind speed - at the height of wheel hub (˚)",
            "Air temperature (°C) ",
            "Power (MW)",
        ]
        self.column_train = [
            "Wind speed at height of 10 meters (m/s)",
            "Wind direction at height of 10 meters (˚)",
            "Wind speed at height of 30 meters (m/s)",
            "Wind direction at height of 30 meters (˚)",
            "Wind speed at height of 50 meters (m/s)",
            "Wind direction at height of 50 meters (˚)",
            "Wind speed - at the height of wheel hub (m/s)",
            "Wind speed - at the height of wheel hub (˚)",
            "Air temperature (°C) ",
            "Power (MW)",
        ]
        self.granularity = 15
        self.granularity_unit = "minute"
        self.urls = [
            "https://raw.githubusercontent.com/Bob05757/Renewable-energy-generation-input-feature-variables-analysis/refs/heads/main/data_processed/wind_farms/Wind farm site 1 (Nominal capacity-99MW).xlsx",
            "https://raw.githubusercontent.com/Bob05757/Renewable-energy-generation-input-feature-variables-analysis/refs/heads/main/data_processed/wind_farms/Wind farm site 2 (Nominal capacity-200MW).xlsx",
            "https://raw.githubusercontent.com/Bob05757/Renewable-energy-generation-input-feature-variables-analysis/refs/heads/main/data_processed/wind_farms/Wind farm site 3 (Nominal capacity-99MW).xlsx",
            "https://raw.githubusercontent.com/Bob05757/Renewable-energy-generation-input-feature-variables-analysis/refs/heads/main/data_processed/wind_farms/Wind farm site 4 (Nominal capacity-66MW).xlsx",
            "https://raw.githubusercontent.com/Bob05757/Renewable-energy-generation-input-feature-variables-analysis/refs/heads/main/data_processed/wind_farms/Wind farm site 5 (Nominal capacity-36MW).xlsx",
            "https://raw.githubusercontent.com/Bob05757/Renewable-energy-generation-input-feature-variables-analysis/refs/heads/main/data_processed/wind_farms/Wind farm site 6 (Nominal capacity-96MW).xlsx",
        ]

    def download(self):
        # Create the directory if it doesn't exist
        os.makedirs(self.path_raw, exist_ok=True)
        os.makedirs(self.path_temp, exist_ok=True)

        for url in self.urls:
            # Download the file
            file_name = url.split("/")[-1]
            file_path = os.path.join(self.path_temp, file_name)
            self.download_file(url=url, save_path=file_path)

            # Read the Excel file
            df = pl.read_excel(file_path)
            df = df.rename(
                {col: col.replace("  ", " ") for col in df.columns if "  " in col}
            )

            # Save as CSV
            df.write_csv(
                os.path.join(self.path_raw, file_name.replace(".xlsx", ".csv"))
            )
