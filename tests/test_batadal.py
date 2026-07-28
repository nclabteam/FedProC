import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import call, patch

import polars as pl

from data_factory.BATADAL import BATADAL


class TestBATADAL(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        temporary_path = Path(self.temporary_directory.name)
        configs = Namespace(
            input_len=2,
            output_len=1,
            offset_len=0,
            train_ratio=0.7,
        )
        self.dataset = BATADAL(configs)
        self.dataset.save_path = str(temporary_path / "BATADAL")
        self.dataset.path_raw = str(
            temporary_path / "BATADAL" / "raw"
        )
        self.dataset.path_temp = str(
            temporary_path / "BATADAL" / "temp"
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_release(
        self,
        path: Path,
        timestamp: str,
        spaced_header: bool = False,
        attack_flag: int | None = None,
    ) -> None:
        columns = ["DATETIME"] + self.dataset.measurement_columns
        values = [timestamp] + list(
            range(1, len(self.dataset.measurement_columns) + 1)
        )
        if attack_flag is not None:
            columns.append("ATT_FLAG")
            values.append(attack_flag)
        if spaced_header:
            columns = [
                column if column == "DATETIME" else f" {column}"
                for column in columns
            ]
        path.write_text(
            ",".join(columns)
            + "\n"
            + ",".join(str(value) for value in values)
            + "\n",
            encoding="utf-8",
        )

    def test_configuration_uses_forty_three_hourly_channels(
        self,
    ) -> None:
        self.assertEqual(self.dataset.granularity, 1)
        self.assertEqual(self.dataset.granularity_unit, "hour")
        self.assertEqual(len(self.dataset.column_target), 43)
        self.assertNotIn("ATT_FLAG", self.dataset.column_target)

    def test_download_merges_releases_by_real_timestamp(self) -> None:
        source_dir = Path(self.dataset.path_temp)
        source_dir.mkdir(parents=True)
        dataset03_path = source_dir / "BATADAL_dataset03.csv"
        dataset04_path = source_dir / "BATADAL_dataset04.csv"
        test_path = source_dir / "BATADAL_test_dataset.csv"
        self._write_release(
            dataset03_path,
            "13/10/14 00",
            attack_flag=0,
        )
        self._write_release(
            dataset04_path,
            "25/12/16 00",
            spaced_header=True,
            attack_flag=-999,
        )
        self._write_release(
            test_path,
            "04/01/17 00",
        )

        with (
            patch.object(
                self.dataset,
                "download_file",
            ) as download,
            patch.object(
                self.dataset,
                "download_and_extract",
            ) as extract,
        ):
            self.dataset.download()

        self.assertEqual(
            download.call_args_list,
            [
                call(
                    url=self.dataset.dataset_urls["dataset03"],
                    save_path=str(dataset03_path),
                ),
                call(
                    url=self.dataset.dataset_urls["dataset04"],
                    save_path=str(dataset04_path),
                ),
            ],
        )
        extract.assert_called_once_with(
            url=self.dataset.dataset_urls["test"],
            save_path=self.dataset.path_temp,
        )
        output = pl.read_csv(
            Path(self.dataset.path_raw) / "BATADAL.csv",
            try_parse_dates=True,
        )
        self.assertEqual(output.shape, (3, 44))
        self.assertNotIn("ATT_FLAG", output.columns)
        self.assertEqual(output["DateTime"][0].year, 2014)
        self.assertEqual(output["DateTime"][1].month, 12)
        self.assertEqual(output["DateTime"][2].month, 1)
        self.assertGreater(
            output["DateTime"][1] - output["DateTime"][0],
            output["DateTime"][2] - output["DateTime"][1],
        )


if __name__ == "__main__":
    unittest.main()
