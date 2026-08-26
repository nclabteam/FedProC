import datetime
import tempfile
import unittest
import zipfile
from argparse import Namespace
from pathlib import Path

import polars as pl

from data_factory.Anemometer import (
    Anemometer,
    AnemometerPaired,
    AnemometerShear,
    AnemometerShear3,
    AnemometerShear4,
)


class TestAnemometer(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / "Anemometer"
        self.configs = Namespace(
            input_len=2,
            output_len=1,
            offset_len=0,
            train_ratio=0.7,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def configure(self, dataset, name: str) -> None:
        dataset.save_path = str(self.root / name)
        dataset.path_raw = str(self.root / name / "raw")
        dataset.path_temp = str(self.root / "temp")
        dataset.path_prepared = str(self.root / name / ".raw_ready")

    def test_download_explains_manual_archive_placement(self) -> None:
        dataset = AnemometerPaired(self.configs)
        self.configure(dataset, "Paired")

        with self.assertRaises(FileNotFoundError) as context:
            dataset.download()

        self.assertIn("pair.zip", str(context.exception))
        self.assertIn(dataset.url, str(context.exception))

    def test_prepare_paired_adds_synthetic_timestamps(self) -> None:
        dataset = AnemometerPaired(self.configs)
        self.configure(dataset, "Paired")
        source_dir = Path(dataset.path_temp)
        source_dir.mkdir(parents=True)
        row = ",".join(str(index) for index in range(16))
        with zipfile.ZipFile(source_dir / "pair.zip", "w") as archive:
            archive.writestr("pair1.txt", f"{row}\n{row}\n")

        dataset.prepare_raw()

        output = pl.read_csv(
            Path(dataset.path_raw) / "pair1.csv",
            try_parse_dates=True,
        )
        self.assertEqual(output.shape, (2, 17))
        self.assertEqual(
            output["Date"][1] - output["Date"][0],
            datetime.timedelta(minutes=10),
        )

    def test_prepare_shear_splits_three_and_four_sensor_files(
        self,
    ) -> None:
        source_dir = self.root / "temp"
        source_dir.mkdir(parents=True)
        date = "06/30/2003 09:50:00"
        shear3 = ",".join(["49", "39", "30"] + ["1"] * 20 + [date])
        shear4 = ",".join(["49", "39", "30", "10"] + ["1"] * 24 + ["2/23/2007 14:10"])
        with zipfile.ZipFile(source_dir / "shear.zip", "w") as archive:
            archive.writestr("shear3.txt", shear3)
            archive.writestr("shear4.txt", shear4)

        dataset3 = AnemometerShear3(self.configs)
        dataset4 = AnemometerShear4(self.configs)
        self.configure(dataset3, "Shear3")
        self.configure(dataset4, "Shear4")
        dataset3.prepare_raw()
        dataset4.prepare_raw()

        output3 = pl.read_csv(
            Path(dataset3.path_raw) / "shear3.csv",
            try_parse_dates=True,
        )
        output4 = pl.read_csv(
            Path(dataset4.path_raw) / "shear4.csv",
            try_parse_dates=True,
        )
        self.assertEqual(output3.shape, (1, 21))
        self.assertEqual(output4.shape, (1, 25))
        self.assertFalse((Path(dataset3.path_raw) / "shear4.csv").exists())
        self.assertFalse((Path(dataset4.path_raw) / "shear3.csv").exists())

    def test_merged_datasets_reference_homogeneous_loaders(self) -> None:
        shear = AnemometerShear(self.configs)
        merged = Anemometer(self.configs)

        self.assertEqual(
            [entry["dataset"] for entry in shear.sets],
            [AnemometerShear3, AnemometerShear4],
        )
        self.assertEqual(
            [entry["dataset"] for entry in merged.sets],
            [
                AnemometerPaired,
                AnemometerShear3,
                AnemometerShear4,
            ],
        )


if __name__ == "__main__":
    unittest.main()
