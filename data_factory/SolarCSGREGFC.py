import os

import polars as pl

from .base import BaseDataset


class SolarCSGREGFC(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set the dataset name and path
        self.save_path = os.path.join("datasets", "SolarCSGREGFC")
        self.path_raw = os.path.join("datasets", "SolarCSGREGFC", "raw")
        self.path_temp = os.path.join("datasets", "SolarCSGREGFC", "temp")

        # Set the dataset parameters
        self.column_date = "Time(year-month-day h:m:s)"
        self.column_target = [
            "Total solar irradiance (W/m2)",
            "Direct normal irradiance (W/m2)",
            "Global horizontal irradiance (W/m2)",
            "Atmosphere (hpa)",
            "Power (MW)",
        ]
        self.column_train = [
            "Total solar irradiance (W/m2)",
            "Direct normal irradiance (W/m2)",
            "Global horizontal irradiance (W/m2)",
            "Atmosphere (hpa)",
            "Power (MW)",
        ]
        self.granularity = 15
        self.granularity_unit = "minute"
        self.urls = [
            "https://raw.githubusercontent.com/Bob05757/Renewable-energy-generation-input-feature-variables-analysis/refs/heads/main/data_processed/solar_stations/Solar station site 1 (Nominal capacity-50MW).xlsx",
            "https://raw.githubusercontent.com/Bob05757/Renewable-energy-generation-input-feature-variables-analysis/refs/heads/main/data_processed/solar_stations/Solar station site 2 (Nominal capacity-130MW).xlsx",
            "https://raw.githubusercontent.com/Bob05757/Renewable-energy-generation-input-feature-variables-analysis/refs/heads/main/data_processed/solar_stations/Solar station site 3 (Nominal capacity-30MW).xlsx",
            "https://raw.githubusercontent.com/Bob05757/Renewable-energy-generation-input-feature-variables-analysis/refs/heads/main/data_processed/solar_stations/Solar station site 4 (Nominal capacity-130MW).xlsx",
            "https://raw.githubusercontent.com/Bob05757/Renewable-energy-generation-input-feature-variables-analysis/refs/heads/main/data_processed/solar_stations/Solar station site 5 (Nominal capacity-110MW).xlsx",
            "https://raw.githubusercontent.com/Bob05757/Renewable-energy-generation-input-feature-variables-analysis/refs/heads/main/data_processed/solar_stations/Solar station site 6 (Nominal capacity-35MW).xlsx",
            "https://raw.githubusercontent.com/Bob05757/Renewable-energy-generation-input-feature-variables-analysis/refs/heads/main/data_processed/solar_stations/Solar station site 7 (Nominal capacity-30MW).xlsx",
            "https://raw.githubusercontent.com/Bob05757/Renewable-energy-generation-input-feature-variables-analysis/refs/heads/main/data_processed/solar_stations/Solar station site 8 (Nominal capacity-30MW).xlsx",
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

            # Save as CSV
            name = file_name.replace(".xlsx", ".csv")
            df.write_csv(os.path.join(self.path_raw, name))
