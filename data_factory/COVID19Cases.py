import os

import polars as pl

from .base import BaseDataset


class COVID19Cases(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = os.path.join("datasets", "COVID19Cases")
        self.path_raw = os.path.join("datasets", "COVID19Cases", "raw")
        self.path_temp = os.path.join("datasets", "COVID19Cases", "temp")
        self.column_date = "date"
        self.column_target = [
            "cases",
            "deaths",
            "fully_vaccinated",
            "partially_vaccinated",
            "dailynewcases",
            "dailynewdeaths",
            "lndailynewcases",
            "lndailynewdeaths",
            "new_full_vaccination",
            "ln_new_full_vaccination",
        ]
        self.column_train = [
            "cases",
            "deaths",
            "fully_vaccinated",
            "partially_vaccinated",
            "dailynewcases",
            "dailynewdeaths",
            "lndailynewcases",
            "lndailynewdeaths",
            "new_full_vaccination",
            "ln_new_full_vaccination",
        ]
        self.segmentation = "state"
        self.granularity = 1
        self.granularity_unit = "day"
        self.url = "https://raw.githubusercontent.com/ashfarhangi/COVID-19/main/data/COVID19_cases.xlsx"

    def download(self):
        os.makedirs(self.path_raw, exist_ok=True)
        os.makedirs(self.path_temp, exist_ok=True)

        file_name = self.url.split("/")[-1]
        file_path = os.path.join(self.path_temp, file_name)
        self.download_file(url=self.url, save_path=file_path)

        df = pl.read_excel(file_path)
        for state in df[self.segmentation].unique().to_list():
            sdf = df.filter(df[self.segmentation] == state)
            sdf = sdf.drop(self.segmentation)
            sdf.write_csv(os.path.join(self.path_raw, f"{state}.csv"))
