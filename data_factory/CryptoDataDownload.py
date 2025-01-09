import datetime
import os

import polars as pl
import requests
from rich.progress import track

from .base import BaseDataset


class CryptoDataDownloadDay(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_raw = os.path.join(
            "datasets", "CryptoDataDownload", "raw", "Binance_Spot_Day"
        )
        self.save_path = os.path.join(
            "datasets", "CryptoDataDownload", "Binance_Spot_Day"
        )
        self.column_date = "Date"
        self.column_target = ["Open", "High", "Low", "Close"]
        self.column_train = ["Open", "High", "Low", "Close"]
        self.granularity = 1
        self.granularity_unit = "day"
        self.url_template = (
            "https://www.cryptodatadownload.com/cdd/Binance_{symbol}_d.csv"
        )

    def get_tickers(self):
        # Define the URL and headers
        url = "https://api.cryptodatadownload.com/v1/data/ohlc/binance/all/available/"
        headers = {"accept": "application/json"}
        # Send the GET request
        response = requests.get(url, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
        else:
            print(f"Failed to retrieve data: {response.status_code}")
            exit()

        return data["result"]["spot"]

    def download(self):
        # Create directories
        os.makedirs(self.path_raw, exist_ok=True)

        # Download data for each ticker
        tickers = self.get_tickers()
        for symbol in track(tickers, description="Downloading data..."):
            # Daily data
            day_url = self.url_template.format(symbol=symbol)
            day_save_path = os.path.join(self.path_raw, f"Binance_{symbol}_d.csv")
            self.download_file(url=day_url, save_path=day_save_path)

    def read(self, path):
        try:
            df = pl.read_csv(path, try_parse_dates=True, skip_rows=1)
            df = self.reduce_polars_df(df=df, info=True)
            return df
        except pl.exceptions.NoDataError:
            print(f"Empty file: {path}")
            return None
        except pl.exceptions.ComputeError:
            print(f"Date have more than one format: {path}")


class CryptoDataDownloadHour(CryptoDataDownloadDay, BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_raw = os.path.join(
            "datasets", "CryptoDataDownload", "raw", "Binance_Spot_Hour"
        )
        self.save_path = os.path.join(
            "datasets", "CryptoDataDownload", "Binance_Spot_Hour"
        )
        self.column_date = "Date"
        self.column_target = ["Open", "High", "Low", "Close"]
        self.column_train = ["Open", "High", "Low", "Close"]
        self.granularity = 1
        self.granularity_unit = "hour"
        self.url_template = (
            "https://www.cryptodatadownload.com/cdd/Binance_{symbol}_1h.csv"
        )


class CryptoDataDownloadMinute(CryptoDataDownloadDay, BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_raw = os.path.join(
            "datasets", "CryptoDataDownload", "raw", "Binance_Spot_Minute"
        )
        self.save_path = os.path.join(
            "datasets", "CryptoDataDownload", "Binance_Spot_Minute"
        )
        self.column_date = "Date"
        self.column_target = ["Open", "High", "Low", "Close"]
        self.column_train = ["Open", "High", "Low", "Close"]
        self.granularity = 1
        self.granularity_unit = "minute"
        self.url_template = (
            "https://www.cryptodatadownload.com/cdd/Binance_{symbol}_{year}_minute.csv"
        )

    def download(self):
        # Create directories
        os.makedirs(self.path_raw, exist_ok=True)

        # Download data for each ticker
        tickers = self.get_tickers()
        for symbol in track(tickers, description="Downloading data..."):
            # Minute-level data by year
            for year in range(2000, datetime.date.today().year + 1):
                minute_url = self.url_template.format(symbol=symbol, year=year)
                minute_save_path = os.path.join(
                    self.path_raw, f"Binance_{symbol}_{year}_minute.csv"
                )

                # Check if the file exists on the server
                response = requests.head(minute_url)
                if response.status_code == 200:
                    self.download_file(url=minute_url, save_path=minute_save_path)

    def read(self, path):
        try:
            df = pl.read_csv(path, try_parse_dates=True, skip_rows=1)
            df = df.rename({col: col.capitalize() for col in df.columns})
            df = self.reduce_polars_df(df=df, info=True)
            return df
        except pl.exceptions.NoDataError:
            print(f"Empty file: {path}")
            return None
        except pl.exceptions.ComputeError:
            print(f"Date have more than one format: {path}")
