import datetime
import os

import polars as pl

from .base import BaseDataset, CustomDataset


class M4Yearly(BaseDataset):
    """M4 competition, Yearly frequency: one series (row) = one federated client.

    M4 series have no real timestamps, just a plain observation index, so a
    synthetic calendar axis is generated at the frequency's own granularity
    purely to satisfy the base pipeline's date-indexed windowing/split logic
    (same trick as PeMS03/04/07/08, which face the same "readings, no real
    dates" problem). Official train.csv + test.csv (the held-out forecast
    horizon) are concatenated in order into one continuous per-series file;
    the framework then re-splits it via train_ratio like every other dataset
    here, so the M4-official train/test boundary is not preserved as-is.

    Yearly/Quarterly/Monthly/.../Hourly only differ by `m4_name`/`m4_unit`
    below (class attributes, resolved per-subclass through MRO) -- all
    download/parsing logic lives here once.
    """

    m4_name = "Yearly"
    m4_granularity = 1
    m4_unit = "year"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = os.path.join("datasets", "M4", self.m4_name)
        self.path_raw = os.path.join("datasets", "M4", self.m4_name, "raw")
        self.path_temp = os.path.join("datasets", "M4", self.m4_name, "temp")

        self.column_date = "date"
        self.column_target = ["Value"]
        self.column_train = ["Value"]
        self.granularity = self.m4_granularity
        self.granularity_unit = self.m4_unit
        # Synthetic dates are gap-free by construction; skip base's fill_date
        # (also sidesteps untested polars "1y"/"1q" interval parsing).
        self.skip_fill_date = True

        base_url = (
            "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset"
        )
        self.url_train = f"{base_url}/Train/{self.m4_name}-train.csv"
        self.url_test = f"{base_url}/Test/{self.m4_name}-test.csv"

    epoch = datetime.datetime(2000, 1, 1)

    def _advance(self, dt: datetime.datetime) -> datetime.datetime:
        unit = self.granularity_unit
        step = self.granularity
        if unit == "year":
            return dt.replace(year=dt.year + step)
        if unit == "quarter":
            months = dt.month - 1 + 3 * step
            return dt.replace(year=dt.year + months // 12, month=months % 12 + 1)
        if unit == "month":
            months = dt.month - 1 + step
            return dt.replace(year=dt.year + months // 12, month=months % 12 + 1)
        if unit == "week":
            return dt + datetime.timedelta(weeks=step)
        if unit == "day":
            return dt + datetime.timedelta(days=step)
        if unit == "hour":
            return dt + datetime.timedelta(hours=step)
        raise ValueError(f"Unsupported M4 granularity_unit: {unit!r}")

    @staticmethod
    def _series_values(row: dict) -> list[float]:
        # M4 CSVs pad short series with trailing blank cells up to the
        # widest series in the file; V1 is the series id, not a value.
        values = []
        for key, val in row.items():
            if key == "V1" or val is None:
                continue
            val = val.strip()
            if val == "":
                continue
            values.append(float(val))
        return values

    def download(self):
        os.makedirs(self.path_raw, exist_ok=True)
        os.makedirs(self.path_temp, exist_ok=True)

        train_path = os.path.join(self.path_temp, f"{self.m4_name}-train.csv")
        test_path = os.path.join(self.path_temp, f"{self.m4_name}-test.csv")
        self.download_file(url=self.url_train, save_path=train_path)
        self.download_file(url=self.url_test, save_path=test_path)

        # infer_schema_length=0 forces Utf8 columns: M4's padded blank cells
        # otherwise trip up polars' numeric type inference.
        train_df = pl.read_csv(train_path, infer_schema_length=0)
        test_df = pl.read_csv(test_path, infer_schema_length=0)
        test_by_id = {row["V1"]: row for row in test_df.iter_rows(named=True)}

        for row in train_df.iter_rows(named=True):
            series_id = row["V1"]
            values = self._series_values(row)
            test_row = test_by_id.get(series_id)
            if test_row is not None:
                values += self._series_values(test_row)
            if len(values) < 10:
                continue

            dt = self.epoch
            dates = []
            for _ in values:
                dates.append(dt)
                dt = self._advance(dt)

            pl.DataFrame({self.column_date: dates, "Value": values}).write_csv(
                os.path.join(self.path_raw, f"{series_id}.csv")
            )

    @staticmethod
    def extract_time_features(dates: pl.Series, freq: str) -> pl.DataFrame:
        # freq_mapping["y"] in BaseDataset is deliberately empty (no useful
        # sub-year signal at yearly cadence), but a 0-column mark tensor
        # divides by zero downstream in split_x_y. Yearly is the only M4
        # frequency that hits this, so patch it locally with a harmless
        # constant column instead of touching base.py for every dataset.
        if freq == "y":
            return pl.DataFrame({"year_marker": [0] * len(dates)})
        return BaseDataset.extract_time_features(dates, freq)


class M4Quarterly(M4Yearly):
    m4_name = "Quarterly"
    m4_unit = "quarter"


class M4Monthly(M4Yearly):
    m4_name = "Monthly"
    m4_unit = "month"


class M4Weekly(M4Yearly):
    m4_name = "Weekly"
    m4_unit = "week"


class M4Daily(M4Yearly):
    m4_name = "Daily"
    m4_unit = "day"


class M4Hourly(M4Yearly):
    m4_name = "Hourly"
    m4_unit = "hour"


class M4(CustomDataset):
    """All six M4 frequencies pooled into one federated benchmark.

    Every series across Yearly/Quarterly/Monthly/Weekly/Daily/Hourly becomes
    its own client (same "row = client" rule as the individual frequency
    datasets); this class only merges their already-generated `info` lists.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = os.path.join("datasets", "M4", "Merged")
        self.sets = [
            {"dataset": M4Yearly},
            {"dataset": M4Quarterly},
            {"dataset": M4Monthly},
            {"dataset": M4Weekly},
            {"dataset": M4Daily},
            {"dataset": M4Hourly},
        ]
