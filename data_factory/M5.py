import csv
import os
from pathlib import Path

from .base import BaseDataset


class M5(BaseDataset):
    """M5 competition sales, with one item-store series per client."""

    url = "https://www.kaggle.com/competitions/m5-forecasting-accuracy"
    calendar_filename = "calendar.csv"
    evaluation_filename = "sales_train_evaluation.csv"
    validation_filename = "sales_train_validation.csv"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_path = os.path.join("datasets", "M5")
        self.path_raw = os.path.join(self.save_path, "raw")
        self.path_temp = os.path.join(self.save_path, "temp")
        self.path_prepared = os.path.join(self.save_path, ".raw_ready")

        self.column_date = "Date"
        self.column_target = ["Value"]
        self.column_train = ["Value"]
        self.granularity = 1
        self.granularity_unit = "day"
        self.skip_fill_date = True

    def download(self) -> None:
        """Tell users how to obtain M5; Kaggle downloads stay manual."""
        os.makedirs(self.path_temp, exist_ok=True)
        raise FileNotFoundError(
            "M5 must be downloaded manually from "
            f"{self.url}. Extract calendar.csv and either "
            "sales_train_evaluation.csv (preferred) or "
            f"sales_train_validation.csv into '{self.path_temp}', then rerun."
        )

    def _sales_path(self) -> Path | None:
        source_dir = Path(self.path_temp)
        for filename in (
            self.evaluation_filename,
            self.validation_filename,
        ):
            path = source_dir / filename
            if path.is_file():
                return path
        return None

    def _load_dates(self, day_names: list[str]) -> list[str]:
        calendar_path = Path(self.path_temp) / self.calendar_filename
        dates_by_day: dict[str, str] = {}

        with calendar_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            for row in csv.DictReader(handle):
                dates_by_day[row["d"]] = row["date"]

        missing_days = [day for day in day_names if day not in dates_by_day]
        if missing_days:
            preview = ", ".join(missing_days[:5])
            raise ValueError(
                f"{calendar_path} is missing M5 day labels: {preview}"
            )
        return [dates_by_day[day] for day in day_names]

    def prepare_raw(self) -> None:
        """Convert the wide Kaggle sales file into per-client CSV files."""
        calendar_path = Path(self.path_temp) / self.calendar_filename
        sales_path = self._sales_path()
        if not calendar_path.is_file() or sales_path is None:
            self.download()

        raw_dir = Path(self.path_raw)
        raw_dir.mkdir(parents=True, exist_ok=True)

        with sales_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as source:
            reader = csv.reader(source)
            header = next(reader)
            id_index = header.index("id")
            day_indices = [
                index
                for index, column in enumerate(header)
                if column.startswith("d_")
            ]
            if not day_indices:
                raise ValueError(f"No M5 day columns found in {sales_path}")

            day_names = [header[index] for index in day_indices]
            dates = self._load_dates(day_names)

            for row in reader:
                client_id = row[id_index]
                output_path = raw_dir / f"{client_id}.csv"
                if output_path.is_file():
                    continue

                with output_path.open(
                    "w",
                    encoding="utf-8",
                    newline="",
                ) as output:
                    writer = csv.writer(output)
                    writer.writerow([self.column_date, "Value"])
                    writer.writerows(
                        (date, row[index])
                        for date, index in zip(dates, day_indices)
                    )

        Path(self.path_prepared).touch()

    def execute(self) -> None:
        if not Path(self.path_prepared).is_file():
            self.prepare_raw()
        super().execute()
