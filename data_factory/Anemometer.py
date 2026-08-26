import csv
import datetime
import itertools
import math
import os
import zipfile
from io import TextIOWrapper
from pathlib import Path
from typing import Iterator

from .base import BaseDataset, CustomDataset

SOURCE_URL = (
    "https://phmsociety.org/phm_competition/"
    "2011-phm-society-conference-data-challenge/"
)
STATISTIC_NAMES = ["Mean", "Std", "Max", "Min"]
ENVIRONMENT_COLUMNS = [f"WindDirection{name}" for name in STATISTIC_NAMES] + [
    f"Temperature{name}" for name in STATISTIC_NAMES
]


def _measurement_columns(sensor_count: int) -> list[str]:
    columns = []
    for sensor in range(1, sensor_count + 1):
        columns.extend(f"Anemometer{sensor}{name}" for name in STATISTIC_NAMES)
    return columns + list(ENVIRONMENT_COLUMNS)


def _manual_download(path_temp: str, archive_name: str) -> None:
    os.makedirs(path_temp, exist_ok=True)
    raise FileNotFoundError(
        f"Download {archive_name} from {SOURCE_URL} and place it in "
        f"'{path_temp}', then rerun."
    )


def _values(row: list[str]) -> list[float | None]:
    values = []
    for raw_value in row:
        value = float(raw_value.strip())
        values.append(value if math.isfinite(value) else None)
    return values


def _parse_date(raw_date: str) -> datetime.datetime:
    for date_format in (
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
    ):
        try:
            return datetime.datetime.strptime(
                raw_date.strip(),
                date_format,
            )
        except ValueError:
            continue
    raise ValueError(f"Unsupported anemometer date: {raw_date!r}")


def _write_client(
    output_path: Path,
    columns: list[str],
    rows: Iterator[list[object]],
) -> None:
    temporary_path = output_path.with_suffix(".tmp")
    with temporary_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output:
        writer = csv.writer(output)
        writer.writerow(["Date"] + columns)
        writer.writerows(rows)
    os.replace(temporary_path, output_path)


class AnemometerPaired(BaseDataset):
    """PHM 2011 paired-anemometer files, one file per client."""

    archive_name = "pair.zip"
    measurement_columns = _measurement_columns(sensor_count=2)
    epoch = datetime.datetime(2000, 1, 1)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        root = os.path.join("datasets", "Anemometer")
        self.save_path = os.path.join(root, "Paired")
        self.path_raw = os.path.join(self.save_path, "raw")
        self.path_temp = os.path.join(root, "temp")
        self.path_prepared = os.path.join(
            self.save_path,
            ".raw_ready",
        )

        self.column_date = "Date"
        self.column_target = list(self.measurement_columns)
        self.column_train = list(self.measurement_columns)
        self.granularity = 10
        self.granularity_unit = "minute"
        self.skip_fill_date = True
        self.url = SOURCE_URL

    def download(self) -> None:
        """Raise with instructions for obtaining the manual archive."""
        _manual_download(
            path_temp=self.path_temp,
            archive_name=self.archive_name,
        )

    def prepare_raw(self) -> None:
        """Convert paired-anemometer text files to per-client CSV files."""
        archive_path = Path(self.path_temp) / self.archive_name
        if not archive_path.is_file():
            self.download()

        raw_dir = Path(self.path_raw)
        raw_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if info.is_dir() or not info.filename.endswith(".txt"):
                    continue

                output_path = raw_dir / f"{Path(info.filename).stem}.csv"
                if output_path.is_file():
                    continue

                with archive.open(info) as source:
                    reader = csv.reader(TextIOWrapper(source))
                    rows = (
                        [
                            self.epoch
                            + datetime.timedelta(minutes=self.granularity * index)
                        ]
                        + _values(row)
                        for index, row in enumerate(reader)
                        if row
                    )
                    _write_client(
                        output_path=output_path,
                        columns=self.measurement_columns,
                        rows=rows,
                    )

        Path(self.path_prepared).touch()

    def execute(self) -> None:
        """Prepare the manual archive before running the base pipeline."""
        if not Path(self.path_prepared).is_file():
            self.prepare_raw()
        super().execute()


class AnemometerShear3(BaseDataset):
    """PHM 2011 three-height shear arrays, one file per client."""

    archive_name = "shear.zip"
    sensor_count = 3
    source_width = 24
    measurement_columns = _measurement_columns(sensor_count)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        root = os.path.join("datasets", "Anemometer")
        self.save_path = os.path.join(root, f"Shear{self.sensor_count}")
        self.path_raw = os.path.join(self.save_path, "raw")
        self.path_temp = os.path.join(root, "temp")
        self.path_prepared = os.path.join(
            self.save_path,
            ".raw_ready",
        )

        self.column_date = "Date"
        self.column_target = list(self.measurement_columns)
        self.column_train = list(self.measurement_columns)
        self.granularity = 10
        self.granularity_unit = "minute"
        self.url = SOURCE_URL

    def download(self) -> None:
        """Raise with instructions for obtaining the manual archive."""
        _manual_download(
            path_temp=self.path_temp,
            archive_name=self.archive_name,
        )

    def _normalized_rows(
        self,
        first_row: list[str],
        reader: Iterator[list[str]],
    ) -> Iterator[list[object]]:
        for row in itertools.chain([first_row], reader):
            if not row:
                continue
            if len(row) != self.source_width:
                raise ValueError(
                    f"Expected {self.source_width} columns, " f"received {len(row)}"
                )
            date = _parse_date(row[-1])
            measurements = row[self.sensor_count : -1]
            yield [date] + _values(measurements)

    def prepare_raw(self) -> None:
        """Convert shear-array text files to per-client CSV files."""
        archive_path = Path(self.path_temp) / self.archive_name
        if not archive_path.is_file():
            self.download()

        raw_dir = Path(self.path_raw)
        raw_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if info.is_dir() or not info.filename.endswith(".txt"):
                    continue

                with archive.open(info) as source:
                    reader = csv.reader(TextIOWrapper(source))
                    first_row = next(reader)
                    if len(first_row) != self.source_width:
                        continue

                    output_path = raw_dir / f"{Path(info.filename).stem}.csv"
                    if output_path.is_file():
                        continue
                    _write_client(
                        output_path=output_path,
                        columns=self.measurement_columns,
                        rows=self._normalized_rows(
                            first_row=first_row,
                            reader=reader,
                        ),
                    )

        Path(self.path_prepared).touch()

    def execute(self) -> None:
        """Prepare the manual archive before running the base pipeline."""
        if not Path(self.path_prepared).is_file():
            self.prepare_raw()
        super().execute()


class AnemometerShear4(AnemometerShear3):
    """PHM 2011 four-height shear arrays, one file per client."""

    sensor_count = 4
    source_width = 29
    measurement_columns = _measurement_columns(sensor_count)


class AnemometerShear(CustomDataset):
    """Merge the homogeneous three- and four-anemometer shear sets."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_path = os.path.join(
            "datasets",
            "Anemometer",
            "Shear",
        )
        self.sets = [
            {"dataset": AnemometerShear3},
            {"dataset": AnemometerShear4},
        ]


class Anemometer(CustomDataset):
    """Merge all paired and shear PHM 2011 anemometer clients."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_path = os.path.join(
            "datasets",
            "Anemometer",
            "Merged",
        )
        self.sets = [
            {"dataset": AnemometerPaired},
            {"dataset": AnemometerShear3},
            {"dataset": AnemometerShear4},
        ]
