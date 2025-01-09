import io
import os
import zipfile

import pandas as pd
import polars as pl
import requests

from .base import BaseDataset


class ElectricityLoadDiagrams(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_raw = os.path.join(
            "datasets", "ElectricityLoadDiagrams", "raw", "LD2011_2014.txt"
        )
        self.save_path = os.path.join("datasets", "ElectricityLoadDiagrams")
        self.column_date = "Date"
        self.column_target = ["Value"]
        self.column_train = ["Value"]
        self.granularity = 15
        self.granularity_unit = "minute"
        self.url = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"

    def download(self):
        dpath = os.path.join(self.save_path, "raw")
        if not os.path.exists(dpath):
            os.makedirs(dpath)
        response = requests.get(self.url)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(dpath)
        os.system(f'rm -r {os.path.join(dpath, "__MACOSX")}')

    def prepossess(self):
        temp_path = os.path.join(self.save_path, "temp")
        if os.path.exists(temp_path):
            return
        os.makedirs(temp_path)
        df = pd.read_csv(self.path_raw, sep=";", decimal=",")
        df = df.rename(columns={df.columns[0]: self.column_date})
        df = pl.DataFrame(df).with_columns(
            pl.col(self.column_date).str.to_datetime("%Y-%m-%d %H:%M:%S")
        )
        cnt = 0
        for col in df.columns:
            if col == self.column_date:
                continue
            sdf = df.select([self.column_date, col])
            sdf = sdf.rename({col: "Value"})
            sdf.write_csv(os.path.join(temp_path, f"{cnt}.csv"))
            cnt += 1
        self.path_raw = temp_path
