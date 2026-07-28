import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import polars as pl

from data_factory.AirQuality import AirQuality


class TestAirQuality(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        temporary_path = Path(self.temporary_directory.name)
        configs = Namespace(
            input_len=2,
            output_len=1,
            offset_len=0,
            train_ratio=0.7,
        )
        self.dataset = AirQuality(configs)
        self.dataset.save_path = str(temporary_path / "AirQuality")
        self.dataset.path_raw = str(
            temporary_path / "AirQuality" / "raw"
        )
        self.dataset.path_temp = str(
            temporary_path / "AirQuality" / "temp"
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_configuration_uses_twelve_hourly_measurements(self) -> None:
        self.assertEqual(self.dataset.granularity, 1)
        self.assertEqual(self.dataset.granularity_unit, "hour")
        self.assertEqual(len(self.dataset.column_target), 12)
        self.assertNotIn("NMHC(GT)", self.dataset.column_target)

    def test_download_normalizes_dates_and_missing_values(self) -> None:
        source_dir = Path(self.dataset.path_temp)
        source_dir.mkdir(parents=True)
        source_path = source_dir / "AirQualityUCI.csv"
        source_path.write_text(
            "Date;Time;CO(GT);PT08.S1(CO);NMHC(GT);"
            "C6H6(GT);PT08.S2(NMHC);NOx(GT);"
            "PT08.S3(NOx);NO2(GT);PT08.S4(NO2);"
            "PT08.S5(O3);T;RH;AH;;\n"
            "10/03/2004;18.00.00;2,6;1360;150;11,9;"
            "1046;166;1056;113;1692;1268;13,6;48,9;"
            "0,7578;;\n"
            "10/03/2004;19.00.00;-200;1292;-200;9,4;"
            "955;103;1174;92;1559;972;13,3;47,7;"
            "0,7255;;\n",
            encoding="utf-8",
        )

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
            Path(self.dataset.path_raw) / "AirQuality.csv",
            try_parse_dates=True,
        )
        self.assertEqual(output.shape, (2, 13))
        self.assertEqual(output["DateTime"][0].year, 2004)
        self.assertEqual(output["CO(GT)"].null_count(), 1)


if __name__ == "__main__":
    unittest.main()
