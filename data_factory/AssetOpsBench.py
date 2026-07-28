import os

import polars as pl

from .base import BaseDataset, CustomDataset


SOURCE_ROOT = (
    "https://raw.githubusercontent.com/IBM/AssetOpsBench/main/"
    "src/couchdb/scenarios_data/shared/iot"
)

CHILLER_COLUMNS = [
    "Chiller 6 Condenser Water Return To Tower Temperature",
    "Chiller 6 Chiller Efficiency",
    "Chiller 6 Tonnage",
    "Chiller 6 Supply Temperature",
    "Chiller 6 Return Temperature",
    "Chiller 6 Condenser Water Flow",
    "Chiller 6 Power Input",
    "Chiller 6 Chiller % Loaded",
    "Chiller 6 Liquid Refrigerant Evaporator Temperature",
    "Chiller 6 Setpoint Temperature",
]

HYDRAULIC_PUMP_COLUMNS = [
    "PS1_Pressure_bar_100Hz",
    "PS2_Pressure_bar_100Hz",
    "PS3_Pressure_bar_100Hz",
    "PS4_Pressure_bar_100Hz",
    "PS5_Pressure_bar_100Hz",
    "PS6_Pressure_bar_100Hz",
    "FS1_VolumeFlow_l_per_min_10Hz",
    "FS2_VolumeFlow_l_per_min_10Hz",
    "TS1_Temperature_C_1Hz",
    "TS2_Temperature_C_1Hz",
    "TS3_Temperature_C_1Hz",
    "TS4_Temperature_C_1Hz",
    "P1_MotorPower_W_100Hz",
    "VS1_Vibration_mm_per_s_1Hz",
    "CE_CoolingEfficiency_percent_1Hz",
    "CP_CoolingPower_kW_1Hz",
    "SE_EfficiencyFactor_percent_1Hz",
]

METRO_PUMP_COLUMNS = [
    "Compressor_Pressure_bar",
    "Pneumatic_Panel_Pressure_bar",
    "Cyclone_Filter_Drop_Pressure_bar",
    "Tower_Discharge_Pressure_Drop_bar",
    "Reservoir_Pressure_bar",
    "Oil_Temperature_C",
    "Motor_Current_A",
    "Air_Intake_Valve_Status",
    "Compressor_Outlet_Valve_Status",
    "Active_Tower_ID",
    "Load_Start_Signal",
    "Low_Pressure_Switch_Status",
    "Tower_Discharge_Switch_Status",
    "Low_Oil_Level_Alarm",
    "Airflow_Pulse_Counter",
]


def _configure_dataset(
    dataset: BaseDataset,
    dataset_name: str,
    source_name: str,
    measurement_columns: list[str],
    granularity: int,
    granularity_unit: str,
) -> None:
    root = os.path.join("datasets", "AssetOpsBench")
    dataset.save_path = os.path.join(root, dataset_name)
    dataset.path_raw = os.path.join(dataset.save_path, "raw")
    dataset.path_temp = os.path.join(root, "temp")

    dataset.column_date = "DateTime"
    dataset.column_target = list(measurement_columns)
    dataset.column_train = list(measurement_columns)
    dataset.granularity = granularity
    dataset.granularity_unit = granularity_unit
    dataset.source_name = source_name
    dataset.url = f"{SOURCE_ROOT}/{source_name}.json"


def _download_dataset(
    dataset: BaseDataset,
    normalize_interval: str | None = None,
) -> None:
    os.makedirs(dataset.path_raw, exist_ok=True)
    os.makedirs(dataset.path_temp, exist_ok=True)
    source_path = os.path.join(
        dataset.path_temp,
        f"{dataset.source_name}.json",
    )
    dataset.download_file(
        url=dataset.url,
        save_path=source_path,
    )

    frame = pl.read_json(
        source_path,
        infer_schema_length=None,
    ).with_columns(
        pl.col("timestamp")
        .str.to_datetime(strict=False)
        .alias(dataset.column_date)
    )
    if normalize_interval is not None:
        frame = (
            frame.with_columns(
                pl.col(dataset.column_date).dt.round(
                    normalize_interval
                )
            )
            .group_by(dataset.column_date)
            .agg(
                [
                    pl.col(column).mean().alias(column)
                    for column in dataset.column_train
                ]
            )
        )

    frame.select(
        [dataset.column_date] + dataset.column_train
    ).drop_nulls().sort(
        dataset.column_date
    ).write_csv(
        os.path.join(
            dataset.path_raw,
            f"{dataset.__class__.__name__}.csv",
        )
    )


class AssetOpsBenchChiller(BaseDataset):
    """AssetOpsBench chiller telemetry normalized to a 15-minute grid."""

    measurement_columns = CHILLER_COLUMNS

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        _configure_dataset(
            self,
            dataset_name="Chiller",
            source_name="chiller_6",
            measurement_columns=self.measurement_columns,
            granularity=15,
            granularity_unit="minute",
        )

    def download(self) -> None:
        """Download and normalize minor timestamp jitter and duplicates."""
        _download_dataset(self, normalize_interval="15m")


class AssetOpsBenchHydraulicPump(BaseDataset):
    """AssetOpsBench daily hydraulic-pump telemetry."""

    measurement_columns = HYDRAULIC_PUMP_COLUMNS

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        _configure_dataset(
            self,
            dataset_name="HydraulicPump",
            source_name="hydraulic_pump_1",
            measurement_columns=self.measurement_columns,
            granularity=1,
            granularity_unit="day",
        )

    def download(self) -> None:
        """Download the telemetry while omitting its cycle index."""
        _download_dataset(self)


class AssetOpsBenchMetroPump(BaseDataset):
    """AssetOpsBench hourly metro-pump telemetry."""

    measurement_columns = METRO_PUMP_COLUMNS

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        _configure_dataset(
            self,
            dataset_name="MetroPump",
            source_name="metro_pump_1",
            measurement_columns=self.measurement_columns,
            granularity=1,
            granularity_unit="hour",
        )

    def download(self) -> None:
        """Download and prepare the metro-pump telemetry."""
        _download_dataset(self)


class AssetOpsBenchMotor(BaseDataset):
    """AssetOpsBench high-frequency motor vibration telemetry."""

    measurement_columns = ["Vibration_X"]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        _configure_dataset(
            self,
            dataset_name="Motor",
            source_name="motor_01",
            measurement_columns=self.measurement_columns,
            granularity=244,
            granularity_unit="microsecond",
        )

    def download(self) -> None:
        """Download and prepare the motor telemetry."""
        _download_dataset(self)


class AssetOpsBench(CustomDataset):
    """Merge the four heterogeneous AssetOpsBench telemetry samples."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_path = os.path.join(
            "datasets",
            "AssetOpsBench",
            "Merged",
        )
        self.sets = [
            {"dataset": AssetOpsBenchChiller},
            {"dataset": AssetOpsBenchHydraulicPump},
            {"dataset": AssetOpsBenchMetroPump},
            {"dataset": AssetOpsBenchMotor},
        ]
