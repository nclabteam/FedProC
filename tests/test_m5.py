import csv
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from data_factory.M5 import M5


class TestM5(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        temporary_path = Path(self.temporary_directory.name)
        configs = Namespace(
            input_len=2,
            output_len=1,
            offset_len=0,
            train_ratio=0.7,
        )
        self.dataset = M5(configs)
        self.dataset.save_path = str(temporary_path / "M5")
        self.dataset.path_raw = str(temporary_path / "M5" / "raw")
        self.dataset.path_temp = str(temporary_path / "M5" / "temp")
        self.dataset.path_prepared = str(
            temporary_path / "M5" / ".raw_ready"
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_download_gives_manual_kaggle_instructions(self) -> None:
        with self.assertRaises(FileNotFoundError) as context:
            self.dataset.download()

        message = str(context.exception)
        self.assertIn(self.dataset.url, message)
        self.assertIn("calendar.csv", message)
        self.assertTrue(Path(self.dataset.path_temp).is_dir())

    def test_prepare_raw_creates_one_client_file_per_series(self) -> None:
        source_dir = Path(self.dataset.path_temp)
        source_dir.mkdir(parents=True)

        with (source_dir / "calendar.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "date",
                    "wm_yr_wk",
                    "weekday",
                    "wday",
                    "month",
                    "year",
                    "d",
                ]
            )
            writer.writerow(
                [
                    "2011-01-29",
                    "11101",
                    "Saturday",
                    "1",
                    "1",
                    "2011",
                    "d_1",
                ]
            )
            writer.writerow(
                [
                    "2011-01-30",
                    "11101",
                    "Sunday",
                    "2",
                    "1",
                    "2011",
                    "d_2",
                ]
            )

        with (source_dir / self.dataset.evaluation_filename).open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "id",
                    "item_id",
                    "dept_id",
                    "cat_id",
                    "store_id",
                    "state_id",
                    "d_1",
                    "d_2",
                ]
            )
            writer.writerow(
                [
                    "FOODS_1_001_CA_1_evaluation",
                    "FOODS_1_001",
                    "FOODS_1",
                    "FOODS",
                    "CA_1",
                    "CA",
                    "2",
                    "3",
                ]
            )

        self.dataset.prepare_raw()

        client_path = (
            Path(self.dataset.path_raw)
            / "FOODS_1_001_CA_1_evaluation.csv"
        )
        self.assertEqual(
            client_path.read_text(encoding="utf-8").splitlines(),
            [
                "Date,Value",
                "2011-01-29,2",
                "2011-01-30,3",
            ],
        )
        self.assertTrue(Path(self.dataset.path_prepared).is_file())


if __name__ == "__main__":
    unittest.main()
