import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import polars as pl

from data_factory.AppliancesEnergy import AppliancesEnergy


class TestAppliancesEnergy(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        temporary_path = Path(self.temporary_directory.name)
        configs = Namespace(
            input_len=2,
            output_len=1,
            offset_len=0,
            train_ratio=0.7,
        )
        self.dataset = AppliancesEnergy(configs)
        self.dataset.save_path = str(temporary_path / "AppliancesEnergy")
        self.dataset.path_raw = str(temporary_path / "AppliancesEnergy" / "raw")
        self.dataset.path_temp = str(temporary_path / "AppliancesEnergy" / "temp")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_configuration_uses_meaningful_measurements(self) -> None:
        self.assertEqual(self.dataset.granularity, 10)
        self.assertEqual(self.dataset.granularity_unit, "minute")
        self.assertEqual(len(self.dataset.column_target), 26)
        self.assertNotIn("rv1", self.dataset.column_target)
        self.assertNotIn("rv2", self.dataset.column_target)

    def test_download_parses_dates_and_removes_noise(self) -> None:
        source_dir = Path(self.dataset.path_temp)
        source_dir.mkdir(parents=True)
        source_path = source_dir / "energydata_complete.csv"
        columns = [
            "date",
            *self.dataset.measurement_columns,
            "rv1",
            "rv2",
        ]
        rows = []
        for minute in (0, 10):
            values = [
                f"2016-01-11 17:{minute:02d}:00",
                *range(1, 27),
                0.25,
                0.75,
            ]
            rows.append(dict(zip(columns, values)))
        pl.DataFrame(rows).write_csv(source_path)

        with patch.object(
            self.dataset,
            "download_and_extract",
        ) as download:
            self.dataset.download()

        download.assert_called_once_with(
            url=self.dataset.url,
            save_path=self.dataset.path_temp,
        )
        output = pl.read_csv(
            Path(self.dataset.path_raw) / "AppliancesEnergy.csv",
            try_parse_dates=True,
        )
        self.assertEqual(output.shape, (2, 27))
        self.assertEqual(output["DateTime"][0].year, 2016)
        self.assertNotIn("rv1", output.columns)
        self.assertNotIn("rv2", output.columns)


if __name__ == "__main__":
    unittest.main()
