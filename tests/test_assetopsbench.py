import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import polars as pl

from data_factory.AssetOpsBench import (
    AssetOpsBench,
    AssetOpsBenchChiller,
    AssetOpsBenchHydraulicPump,
    AssetOpsBenchMetroPump,
    AssetOpsBenchMotor,
)


class TestAssetOpsBench(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / "AssetOpsBench"
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

    def test_subdataset_configurations(self) -> None:
        chiller = AssetOpsBenchChiller(self.configs)
        hydraulic = AssetOpsBenchHydraulicPump(self.configs)
        metro = AssetOpsBenchMetroPump(self.configs)
        motor = AssetOpsBenchMotor(self.configs)

        self.assertEqual(
            (chiller.granularity, chiller.granularity_unit),
            (15, "minute"),
        )
        self.assertEqual(len(chiller.column_target), 10)
        self.assertNotIn("Chiller 6 Schedule", chiller.column_target)
        self.assertEqual(len(hydraulic.column_target), 17)
        self.assertNotIn("cycle", hydraulic.column_target)
        self.assertEqual(len(metro.column_target), 15)
        self.assertEqual(
            (motor.granularity, motor.granularity_unit),
            (244, "microsecond"),
        )

    def test_chiller_download_rounds_and_merges_timestamps(
        self,
    ) -> None:
        dataset = AssetOpsBenchChiller(self.configs)
        self.configure(dataset, "Chiller")
        source_dir = Path(dataset.path_temp)
        source_dir.mkdir(parents=True)
        rows = []
        for timestamp, value in (
            ("2020-06-01T00:00:00", 1.0),
            ("2020-06-01T00:00:03", 3.0),
            ("2020-06-01T00:15:04", 5.0),
        ):
            row = {
                "asset_id": "Chiller 6",
                "timestamp": timestamp,
                "Chiller 6 Schedule": 1,
            }
            row.update({column: value for column in dataset.measurement_columns})
            rows.append(row)
        source_path = source_dir / "chiller_6.json"
        source_path.write_text(
            json.dumps(rows),
            encoding="utf-8",
        )

        with patch.object(dataset, "download_file") as download:
            dataset.download()

        download.assert_called_once_with(
            url=dataset.url,
            save_path=str(source_path),
        )
        output = pl.read_csv(
            Path(dataset.path_raw) / "AssetOpsBenchChiller.csv",
            try_parse_dates=True,
        )
        self.assertEqual(output.shape, (2, 11))
        self.assertEqual(output[dataset.measurement_columns[0]][0], 2.0)
        self.assertNotIn("Chiller 6 Schedule", output.columns)
        self.assertEqual(output["DateTime"][0].minute, 0)
        self.assertEqual(output["DateTime"][1].minute, 15)

    def test_umbrella_references_all_telemetry_samples(self) -> None:
        dataset = AssetOpsBench(self.configs)

        self.assertEqual(
            [entry["dataset"] for entry in dataset.sets],
            [
                AssetOpsBenchChiller,
                AssetOpsBenchHydraulicPump,
                AssetOpsBenchMetroPump,
                AssetOpsBenchMotor,
            ],
        )


if __name__ == "__main__":
    unittest.main()
