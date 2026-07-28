import os

import polars as pl

from .base import BaseDataset


LEVEL_COLUMNS = [f"L_T{index}" for index in range(1, 8)]
PUMP_COLUMNS = [
    column
    for index in range(1, 12)
    for column in (f"F_PU{index}", f"S_PU{index}")
]
VALVE_COLUMNS = ["F_V2", "S_V2"]
PRESSURE_COLUMNS = [
    "P_J280",
    "P_J269",
    "P_J300",
    "P_J256",
    "P_J289",
    "P_J415",
    "P_J302",
    "P_J306",
    "P_J307",
    "P_J317",
    "P_J14",
    "P_J422",
]
MEASUREMENT_COLUMNS = (
    LEVEL_COLUMNS
    + PUMP_COLUMNS
    + VALVE_COLUMNS
    + PRESSURE_COLUMNS
)


class BATADAL(BaseDataset):
    """Hourly SCADA telemetry from the C-Town water network."""

    measurement_columns = MEASUREMENT_COLUMNS

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_path = os.path.join("datasets", "BATADAL")
        self.path_raw = os.path.join(self.save_path, "raw")
        self.path_temp = os.path.join(self.save_path, "temp")

        self.column_date = "DateTime"
        self.column_target = list(self.measurement_columns)
        self.column_train = list(self.measurement_columns)
        self.granularity = 1
        self.granularity_unit = "hour"
        self.url = "https://www.batadal.net/data.html"
        self.dataset_urls = {
            "dataset03": (
                "https://www.batadal.net/data/"
                "BATADAL_dataset03.csv"
            ),
            "dataset04": (
                "https://www.batadal.net/data/"
                "BATADAL_dataset04.csv"
            ),
            "test": (
                "https://www.batadal.net/data/"
                "BATADAL_test_dataset.zip"
            ),
        }

    def _read_release(
        self,
        path: str,
        date_format: str,
    ) -> pl.DataFrame:
        frame = pl.read_csv(
            path,
            infer_schema_length=None,
        )
        frame = frame.rename(
            {
                column: column.strip()
                for column in frame.columns
            }
        )
        return frame.with_columns(
            [
                pl.concat_str(
                    [
                        pl.col("DATETIME").str.strip_chars(),
                        pl.lit(":00"),
                    ]
                )
                .str.to_datetime(f"{date_format}:%M")
                .alias(self.column_date),
                *[
                    pl.col(column)
                    .cast(pl.String)
                    .str.strip_chars()
                    .cast(pl.Float64)
                    .alias(column)
                    for column in self.measurement_columns
                ],
            ]
        ).select(
            [self.column_date] + self.measurement_columns
        )

    def download(self) -> None:
        """Download and merge the three canonical BATADAL releases."""
        os.makedirs(self.path_raw, exist_ok=True)
        os.makedirs(self.path_temp, exist_ok=True)

        dataset03_path = os.path.join(
            self.path_temp,
            "BATADAL_dataset03.csv",
        )
        dataset04_path = os.path.join(
            self.path_temp,
            "BATADAL_dataset04.csv",
        )
        test_path = os.path.join(
            self.path_temp,
            "BATADAL_test_dataset.csv",
        )
        self.download_file(
            url=self.dataset_urls["dataset03"],
            save_path=dataset03_path,
        )
        self.download_file(
            url=self.dataset_urls["dataset04"],
            save_path=dataset04_path,
        )
        self.download_and_extract(
            url=self.dataset_urls["test"],
            save_path=self.path_temp,
        )

        releases = [
            self._read_release(
                dataset03_path,
                "%d/%m/%y %H",
            ),
            self._read_release(
                dataset04_path,
                "%d/%m/%y %H",
            ),
            self._read_release(
                test_path,
                "%d/%m/%y %H",
            ),
        ]
        pl.concat(
            releases,
            how="vertical_relaxed",
        ).unique(
            subset=[self.column_date],
        ).sort(
            self.column_date
        ).write_csv(
            os.path.join(self.path_raw, "BATADAL.csv")
        )
