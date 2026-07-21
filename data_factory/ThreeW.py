import os

import polars as pl
import requests

from .base import BaseDataset, CustomDataset


class ThreeWBase(BaseDataset):
    """Petrobras 3W offshore well dataset: one instance (episode) = one federated client.

    3W organizes real/simulated/hand-drawn recordings into ten class-label
    folders (`dataset/0` .. `dataset/9`, one per fault type incl. "normal").
    Each parquet file is a self-contained episode with its own real
    timestamp column -- unlike M4, no synthetic dates are needed here, and
    `fill_date` is left enabled since real recordings do have internal gaps.

    Filenames are only unique *within* a class folder (e.g. `SIMULATED_00001`
    exists in seven different folders), so raw CSVs are named
    `{label}_{original_stem}.csv` to avoid distinct instances colliding.

    Not every instance carries every sensor: presence is all-or-nothing per
    instance (a well either has a sensor installed for its whole episode or
    not at all), so only the five variables consistently populated across
    wells -- and used as the standard "core" set in 3W literature -- are
    kept as the multivariate target/train columns; instances missing any of
    them are skipped since `prepossess`'s `drop_nulls` would empty them
    anyway.
    """

    name_prefix = None
    dataset_dirname = None
    core_columns = ["P-PDG", "P-TPT", "T-TPT", "P-MON-CKP", "T-JUS-CKP"]
    label_folders = [str(i) for i in range(10)]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = os.path.join("datasets", "3W", self.dataset_dirname)
        self.path_raw = os.path.join("datasets", "3W", self.dataset_dirname, "raw")
        self.path_temp = os.path.join("datasets", "3W", self.dataset_dirname, "temp")

        self.column_date = "date"
        self.column_target = list(self.core_columns)
        self.column_train = list(self.core_columns)
        self.granularity = 1
        self.granularity_unit = "second"

        self.repo_api = "https://api.github.com/repos/petrobras/3W/contents/dataset"
        self.repo_raw = "https://raw.githubusercontent.com/petrobras/3W/main/dataset"

    @staticmethod
    def read(path: str) -> pl.DataFrame | None:
        # Sensors can sit null for a long warm-up run before reporting, so
        # FileManager.read's default (limited-row) schema inference can
        # mistake a numeric column for categorical; scan the whole file.
        try:
            df = pl.read_csv(path, try_parse_dates=True, infer_schema_length=None)
            return df
        except pl.exceptions.NoDataError:
            print(f"Empty file: {path}")
            return None

    def _list_instance_files(self, label: str) -> list[str]:
        response = requests.get(f"{self.repo_api}/{label}", timeout=self.request_timeout)
        response.raise_for_status()
        return [
            entry["name"]
            for entry in response.json()
            if entry["name"].startswith(self.name_prefix)
            and entry["name"].endswith(".parquet")
        ]

    def download(self):
        os.makedirs(self.path_raw, exist_ok=True)
        os.makedirs(self.path_temp, exist_ok=True)

        for label in self.label_folders:
            try:
                filenames = self._list_instance_files(label)
            except Exception as exc:
                print(f"3W: failed to list folder {label}: {exc}")
                continue

            for filename in filenames:
                stem = os.path.splitext(filename)[0]
                tmp_path = os.path.join(self.path_temp, f"{label}_{filename}")
                try:
                    self.download_file(
                        url=f"{self.repo_raw}/{label}/{filename}", save_path=tmp_path
                    )
                    df = pl.read_parquet(tmp_path)
                except Exception as exc:
                    print(f"3W: skipping {label}/{filename}: {exc}")
                    continue
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

                if not all(col in df.columns for col in self.core_columns):
                    continue
                if any(df[col].null_count() == df.height for col in self.core_columns):
                    continue

                df = df.select(["timestamp"] + self.core_columns).rename(
                    {"timestamp": self.column_date}
                )
                df.write_csv(os.path.join(self.path_raw, f"{label}_{stem}.csv"))


class ThreeWReal(ThreeWBase):
    name_prefix = "WELL"
    dataset_dirname = "Real"


class ThreeWSimulated(ThreeWBase):
    name_prefix = "SIMULATED"
    dataset_dirname = "Simulated"


class ThreeW(CustomDataset):
    """Real + simulated 3W instances pooled into one federated benchmark."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = os.path.join("datasets", "3W", "Merged")
        self.sets = [
            {"dataset": ThreeWReal},
            {"dataset": ThreeWSimulated},
        ]
