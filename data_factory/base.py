import json
import os
import re
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import entropy

from .data_frame_optimizer import DataFrameOptimizer
from .file_manager import FileManager
from .time_series_characteristics import TimeSeriesCharacteristics


class BaseDataset(TimeSeriesCharacteristics, FileManager, DataFrameOptimizer):
    """
    Base class for handling time series dataset processing.

    This class provides functionalities for downloading, reading, preprocessing,
    analyzing, splitting, and saving time series data into a format suitable
    for sequence modeling tasks. It handles aspects like memory optimization,
    feature extraction, statistical analysis, and data splitting based on
    configurations.
    """

    def __init__(
        self,
        configs: dict,
    ) -> None:
        """
        Initialize the BaseDataset instance.

        Sets up configuration parameters, sequence lengths, and paths for time series
        dataset processing. Reads skip_fill_date from configs to handle discontinuous
        datasets (e.g., stock market data with no weekends).

        Args:
            configs (dict): Configuration dictionary with keys:
                - input_len (int): Input sequence length
                - output_len (int): Prediction/output sequence length
                - offset_len (int): Gap between input and output sequences
                - train_ratio (float): Training data ratio (0.0-1.0)
                - skip_fill_date (bool, optional): Skip date filling for discontinuous data.
                  Defaults to False.

        Attributes:
            seq_len (int): Input sequence length
            pred_len (int): Prediction sequence length
            offset_len (int): Offset length between sequences
            train_ratio (float): Training data ratio
            skip_fill_date (bool): Whether to skip filling missing dates
            info (list): Metadata information for clients
            path_raw (str | None): Raw data path
            save_path (str | None): Save directory path
            column_date (str | None): Date column name
            column_target (list[str] | None): Target column names
            column_train (list[str] | None): Training column names
            granularity (int | None): Time step granularity
            granularity_unit (str | None): Unit of granularity (minute, hour, day, etc.)
        """
        self.configs = configs
        self.seq_len: int = configs.input_len
        self.pred_len: int = configs.output_len
        self.offset_len: int = configs.offset_len
        self.train_ratio: float = configs.train_ratio
        self.skip_fill_date: bool = False
        self.info: list[dict] = []

        self.path_raw: str | None = None
        self.save_path: str | None = None
        self.column_date: str | None = None
        self.column_target: list[str] | None = None
        self.column_train: list[str] | None = None
        self.granularity: int | None = None
        self.granularity_unit: str | None = None

    @staticmethod
    def convert_granularity_unit(granularity_unit: str) -> str:
        """
        Convert human-readable time granularity units to frequency strings.

        Maps full unit names to their abbreviated frequency strings compatible with
        Polars datetime operations.

        Args:
            granularity_unit (str): Full unit name (e.g., "hour", "day", "minute").

        Returns:
            str: Frequency abbreviation (e.g., "h", "d", "m").
                 Returns original string if no mapping found.

        Examples:
            >>> BaseDataset.convert_granularity_unit("hour")
            'h'
            >>> BaseDataset.convert_granularity_unit("day")
            'd'
        """
        unit_dict = {
            "nanosecond": "ns",
            "microsecond": "us",
            "millisecond": "ms",
            "second": "s",
            "minute": "m",
            "hour": "h",
            "day": "d",
            "week": "w",
            "month": "mo",
            "quarter": "q",
            "year": "y",
        }
        return unit_dict.get(granularity_unit, granularity_unit)

    @staticmethod
    def extract_time_features(dates: pl.Series, freq: str) -> pl.DataFrame:
        """
        Extract normalized time-based features from datetime series.

        Extracts temporal features (hour, day, week, month, year components) based on
        the data frequency. All features are normalized to [-0.5, 0.5] range to prevent
        bias toward larger numerical values. Features are selected based on frequency
        to avoid redundancy (e.g., seconds are irrelevant for daily data).

        Args:
            dates (pl.Series): Polars Series with datetime objects (DateTime or Date type).
            freq (str): Frequency string indicating data granularity:
                - 'y': Year (returns empty features)
                - 'q', 'mo': Quarterly/Monthly (month_of_year)
                - 'w': Weekly (day_of_month, week_of_year)
                - 'd': Daily (hour, day_of_week, day_of_month, day_of_year)
                - 'h': Hourly (hour_of_day, day_of_week, day_of_month, day_of_year)
                - 'm': Minute (minute_of_hour, hour_of_day, day_of_week, day_of_month, day_of_year)
                - 's', 'ms', 'us', 'ns': Sub-second (all features)

        Returns:
            pl.DataFrame: DataFrame with extracted features, each column normalized to [-0.5, 0.5].
                         Columns include: second_of_minute, minute_of_hour, hour_of_day,
                         day_of_week, day_of_month, day_of_year, month_of_year, week_of_year.

        Raises:
            ValueError: If frequency string is not supported.

        Examples:
            >>> dates = pl.datetime_range(datetime(2023, 1, 1), datetime(2023, 1, 31), interval="1d")
            >>> features = BaseDataset.extract_time_features(dates, 'd')
            >>> features.shape
            (31, 4)  # 31 days with 4 daily features
        """

        # Raw integer indices — normalized inside the model when needed.
        # Order: [month(0), day(1), weekday(2), hour(3), minute(4), second(5)]
        # matches TemporalEmbedding column indices.
        features = {
            "second_of_minute": lambda dt: dt.dt.second(),
            "minute_of_hour": lambda dt: dt.dt.minute(),
            "hour_of_day": lambda dt: dt.dt.hour(),
            "day_of_week": lambda dt: dt.dt.weekday(),
            "day_of_month": lambda dt: dt.dt.day(),
            "day_of_year": lambda dt: dt.dt.ordinal_day(),
            "month_of_year": lambda dt: dt.dt.month(),
            "week_of_year": lambda dt: dt.dt.week(),
        }

        # Map frequencies to time features in TemporalEmbedding column order:
        # [month(0), day(1), weekday(2), hour(3), minute(4), second(5)]
        freq_mapping = {
            "y": [],
            "q": ["month_of_year"],
            "mo": ["month_of_year"],
            "w": ["month_of_year", "day_of_month", "week_of_year"],
            "d": ["month_of_year", "day_of_month", "day_of_week"],
            "h": ["month_of_year", "day_of_month", "day_of_week", "hour_of_day"],
            "t": [
                "month_of_year",
                "day_of_month",
                "day_of_week",
                "hour_of_day",
                "minute_of_hour",
            ],
            "m": [
                "month_of_year",
                "day_of_month",
                "day_of_week",
                "hour_of_day",
                "minute_of_hour",
            ],
            "s": [
                "month_of_year",
                "day_of_month",
                "day_of_week",
                "hour_of_day",
                "minute_of_hour",
                "second_of_minute",
            ],
            "ms": [
                "month_of_year",
                "day_of_month",
                "day_of_week",
                "hour_of_day",
                "minute_of_hour",
                "second_of_minute",
            ],
            "us": [
                "month_of_year",
                "day_of_month",
                "day_of_week",
                "hour_of_day",
                "minute_of_hour",
                "second_of_minute",
            ],
            "ns": [
                "month_of_year",
                "day_of_month",
                "day_of_week",
                "hour_of_day",
                "minute_of_hour",
                "second_of_minute",
            ],
        }

        # Validate frequency
        if freq not in freq_mapping:
            raise ValueError(
                f"Unsupported frequency {freq}. Supported: {list(freq_mapping.keys())}"
            )

        # Select and compute features based on the frequency
        selected_features = {
            name: func(dates)
            for name, func in features.items()
            if name in freq_mapping[freq]
        }

        return pl.DataFrame(selected_features)

    @staticmethod
    def subdf_from_indices(
        df: pl.DataFrame,
        date_column: str,
        start: pl.datatypes.Date | pl.datatypes.Datetime,
        end: pl.datatypes.Date | pl.datatypes.Datetime,
    ) -> pl.DataFrame:
        """
        Filter a DataFrame to select rows within a specified date range.

        Applies an inclusive-exclusive temporal range filter [start, end) on the specified
        date/datetime column. This is a core utility for partitioning time series data into
        training/validation/test splits or extracting data for specific temporal periods.

        The filter semantics follow standard interval notation: the start date is included,
        while the end date is excluded. This ensures adjacent time ranges don't overlap and
        cover the entire dataset when concatenated.

        Args:
            df (pl.DataFrame): Input DataFrame containing the date column to filter on.
                              Must have at least one row and the specified date column.
            date_column (str): Name of the date or datetime column to apply the temporal filter.
                              Column must be of type pl.Date, pl.Datetime, or compatible type.
            start (pl.datatypes.Date | pl.datatypes.Datetime): Start boundary (inclusive).
                                                               Can be pl.Date or pl.Datetime type.
                                                               Rows with date_column >= start are included.
            end (pl.datatypes.Date | pl.datatypes.Datetime): End boundary (exclusive).
                                                             Can be pl.Date or pl.Datetime type.
                                                             Rows with date_column < end are included.

        Returns:
            pl.DataFrame: Filtered DataFrame containing only rows where:
                         start <= date_column < end.
                         Preserves all original columns and row order.
                         May be empty if no rows fall within the range.
        """
        return df.filter((pl.col(date_column) >= start) & (pl.col(date_column) < end))

    @staticmethod
    def sliding_window(df: pl.DataFrame, seq_len: int, offset_len: int, pred_len: int):
        """
        Creates lagged features using a sliding window approach.

        For each column in the input DataFrame, it generates new columns representing
        past values (for input sequence) and future values (for prediction sequence),
        skipping the offset period.

        Args:
            df (pl.DataFrame): The input time series DataFrame.
            seq_len (int): The length of the input sequence.
            offset_len (int): The gap between the input and prediction sequences.
            pred_len (int): The length of the prediction sequence.

        Returns:
            pl.DataFrame: A DataFrame with original columns and newly created lagged
                          and future columns. Rows with nulls introduced by shifting
                          are dropped.
        """

        # Total window size needed to create one complete sample (input + offset + prediction)
        lag_window = seq_len + offset_len + pred_len

        # Iterate backwards from the furthest point needed (pred_len ahead)
        # down to the oldest point needed (seq_len-1 behind)
        cols = df.columns
        for i in range(1, lag_window):
            for col in cols:
                if seq_len - 1 < i and i <= (seq_len - 1 + offset_len):
                    continue  # Skip creating columns for the offset gap
                name = f"{col}_ahead_{i}"
                df = df.with_columns(pl.col(col).shift(-i).alias(name))
        # Drop rows at the end where future values couldn't be computed (NaNs)
        df = df.drop_nulls()
        return df

    def dir(self):
        name = f"seq_{self.seq_len}-offset_{self.offset_len}-pred_{self.pred_len}"
        self.path_save = os.path.join(self.save_path, name)
        self.path_info = os.path.join(self.path_save, "info.json")

        if not os.path.exists(self.path_raw):
            raise FileNotFoundError(f"Path '{self.path_raw}' not found")

        self.path_train = os.path.join(self.path_save, "train")
        # self.path_valid = os.path.join(self.path_save, "valid")
        self.path_test = os.path.join(self.path_save, "test")
        for path in [
            self.path_save,
            self.path_train,
            #  self.path_valid,
            self.path_test,
        ]:
            if not os.path.exists(path):
                os.makedirs(path)

    def get_config(self):
        """Constructs the configuration dictionary."""
        return {
            "train_ratio": self.train_ratio,
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "offset_len": self.offset_len,
            "column_date": self.column_date,
            "column_target": self.column_target,
            "column_train": self.column_train,
            "granularity": self.granularity,
            "granularity_unit": self.granularity_unit,
            "input_channels": len(self.column_train),
            "output_channels": len(self.column_target),
        }

    def save_info(self):
        """Saves the configuration to a file."""
        with open(self.path_info, "w") as f:
            json.dump(self.info, f, indent=4)

    def check(self):
        """Checks if the saved configuration matches the current settings, excluding 'clients'."""
        if os.path.exists(self.path_info):
            with open(self.path_info, "r") as f:
                config = json.load(f)

            # Get configurations excluding 'clients' for comparison
            current_config = self.get_config()
            saved_config = {
                k: v for k, v in config[0].items() if k in current_config.keys()
            }

            # Check if the current configuration matches the saved configuration
            if current_config == saved_config:
                self.info = config
                return True
        return False

    def fix_params(self):
        self.granularity_unit = self.convert_granularity_unit(self.granularity_unit)
        self.column_used = list(set(self.column_train) | set(self.column_target))

    def fill_date(
        self,
        df: pl.DataFrame,
        column_date: str,
        start=None,
        end=None,
        granularity: str = "1d",
    ) -> pl.DataFrame:
        # Skip filling dates for discontinuous datasets (e.g., stock data with no weekends)
        if self.skip_fill_date:
            return df

        if not start:
            start = df[column_date].min()
        if not end:
            end = df[column_date].max()
        return df.join(
            other=self.reduce_polars_df(
                df=pl.datetime_range(
                    start=start,
                    end=end,
                    interval=granularity,
                    closed="both",
                    eager=True,
                )
                .alias(self.column_date)
                .to_frame(),
                info=False,
            ),
            on=column_date,
            how="full",
            coalesce=True,
        )

    def get_statistic(self, df: pl.DataFrame):
        statistics = (
            df.select(
                count=pl.struct(pl.all().count()),
                n_unique=pl.struct(pl.all().n_unique()),
                n_null=pl.struct(pl.all().null_count()),
                n_neg=pl.struct((pl.all() < 0).sum()),
                n_zero=pl.struct((pl.all() == 0).sum()),
                n_pos=pl.struct((pl.all() > 0).sum()),
                min=pl.struct(pl.all().min()),
                max=pl.struct(pl.all().max()),
                mean=pl.struct(pl.all().mean()),
                median=pl.struct(pl.all().median()),
                std=pl.struct(pl.all().std()),
                var=pl.struct(pl.all().var()),
                q1=pl.struct(pl.all().quantile(0.25)),
                q3=pl.struct(pl.all().quantile(0.75)),
                iqr=pl.struct(pl.all().quantile(0.75) - pl.all().quantile(0.25)),
                cv=pl.struct(
                    pl.all().std() / (pl.all().mean() + 1e-10)
                ),  # Avoid division by zero
            )
            .melt()
            .unnest("value")
        )

        statistics = pl.concat(
            [
                statistics,
                # self.calculate_shifting_values(df=df),
                self.get_transition_value(df=df),
                self.compute_trend_seasonal_strength_and_entropy_auto(
                    df=df,
                    granularity=self.granularity,
                    granularity_unit=self.granularity_unit,
                ),
            ]
        )

        statistics = statistics.transpose(include_header=True).pipe(
            lambda df_t: (
                df_t.rename(
                    {
                        old_col: (
                            str(new_col)
                            if new_col != df_t.slice(0, 1).select(pl.first()).item()
                            else "col"
                        )
                        for old_col, new_col in zip(
                            df_t.columns, df_t.slice(0, 1).rows()[0]
                        )
                    }
                ).slice(1)
            )
        )

        statistics = {
            item["col"]: {
                key: float(value) for key, value in item.items() if key != "col"
            }
            for item in statistics.to_dicts()
        }

        return statistics

    def split_x_y(self, df, x_used_cols, y_used_cols):
        ori_cols = df.columns
        df = self.sliding_window(
            df=df,
            seq_len=self.seq_len,
            offset_len=self.offset_len,
            pred_len=self.pred_len,
        )
        idx = len(ori_cols) * self.seq_len

        x = df.select(pl.nth(range(0, idx)))
        x_cols = [
            col
            for col in x.columns
            if any(
                col.startswith(target + "_") or col == target for target in x_used_cols
            )
        ]
        x = x.select(x_cols)
        x = x.to_numpy()
        x = x.reshape(x.shape[0], x.shape[1] // len(x_used_cols), len(x_used_cols))

        y = df.select(pl.nth(range(idx, len(df.columns))))
        y_cols = [
            col
            for col in y.columns
            if any(
                col.startswith(target + "_") or col == target for target in y_used_cols
            )
        ]
        y = y.select(y_cols)
        y = y.to_numpy()
        y = y.reshape(y.shape[0], y.shape[1] // len(y_used_cols), len(y_used_cols))

        return x, y

    def correct_indices(self, df):
        dates = df[self.column_date].to_list()
        if self.total_null != 0 and not self.skip_fill_date:
            temp = self.sliding_window(
                df=df,
                seq_len=self.seq_len,
                offset_len=self.offset_len,
                pred_len=self.pred_len,
            )
            start_day_x = temp[self.column_date].to_list()
            split_indices = [0, int(len(start_day_x) * self.train_ratio)]

        else:
            s = len(df)
            split_indices = [0, int(s * self.train_ratio)]
            start_day_x = dates

        # Skip processing if not enough data points
        if len(start_day_x) < 10:
            return []

        res = [start_day_x[idx] for idx in split_indices]
        res.append(dates[-1])

        fres = [[res[0], res[1]]]
        for i in range(1, len(res) - 1):
            fres.append(
                [
                    dates[
                        df.with_row_count()
                        .filter(pl.col(self.column_date) == res[1])
                        .select("row_nr")
                        .to_numpy()
                        .flatten()[0]
                        + 1
                    ],
                    res[i + 1],
                ]
            )
        return fres

    def prepossess(self, df):
        df = self.reduce_polars_df(df=df, info=True)
        df = df.shrink_to_fit().unique().drop_nulls()
        return df

    def get_file_paths(self):
        self.file_pahts_list = [
            os.path.join(self.path_raw, path) for path in os.listdir(self.path_raw)
        ]

        # human sort the file paths
        self.file_pahts_list.sort(
            key=lambda text: [
                int(c) if c.isdigit() else c for c in re.split(r"(\d+)", text)
            ]
        )

    def generate(self):
        for path in self.file_pahts_list:
            start_time = time.time()
            print(f"{path = }")

            # read file
            df = self.read(path)
            if df is None:
                continue

            # prepossess data
            df = self.prepossess(df)

            # sort by date
            df = df.sort(self.column_date)

            # get raw statistic count null
            df = self.fill_date(
                df=df,
                column_date=self.column_date,
                granularity=str(self.granularity) + self.granularity_unit,
            )
            raw_stats = self.get_statistic(df.select(self.column_used))
            self.total_null = sum(raw_stats[column]["n_null"] for column in raw_stats)

            # compute correct split indices
            split_indices = self.correct_indices(
                df.select([self.column_date] + self.column_used)
            )
            if not split_indices:
                continue

            client_id = len(self.info)
            client_info = {
                "client": client_id,
                "file": path,
                "stats": {"raw": raw_stats},
                "paths": {
                    "train": {},
                    "test": {},
                },
                "date": {},
                "samples": {},
                "size_mb": {},
            }
            for split_name, (start, end) in zip(["train", "test"], split_indices):
                split_df = self.subdf_from_indices(
                    df=df, date_column=self.column_date, start=start, end=end
                )
                client_info["date"][split_name] = {
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                }

                # Extract time features
                split_time_df = self.extract_time_features(
                    dates=split_df[self.column_date], freq=self.granularity_unit
                )
                null_mask = split_df.with_columns(
                    pl.concat_list(pl.all().is_null()).alias("null_mask")
                )["null_mask"].list.any()
                split_time_df = split_time_df.with_columns(
                    [
                        pl.when(null_mask)
                        .then(None)
                        .otherwise(split_time_df[col])
                        .alias(col)
                        for col in split_time_df.columns
                    ]
                )

                # Get statistic
                split_df = split_df.select(self.column_used)
                split_stats = self.get_statistic(split_df)

                # Split data
                split_x, split_y = self.split_x_y(
                    df=split_df,
                    x_used_cols=self.column_used,
                    y_used_cols=self.column_target,
                )
                client_info["samples"][split_name] = {
                    "x": str(split_x.shape),
                    "y": str(split_y.shape),
                }
                split_x_mark, split_y_mark = self.split_x_y(
                    df=split_time_df,
                    x_used_cols=split_time_df.columns,
                    y_used_cols=split_time_df.columns,
                )

                # Save split data
                file_path = os.path.join(
                    getattr(self, f"path_{split_name}"), f"{client_id}.npz"
                )
                np.savez_compressed(
                    file=file_path,
                    x_mark=split_x_mark,
                    y_mark=split_y_mark,
                    x=split_x,
                    y=split_y,
                )
                client_info["paths"][split_name] = file_path
                client_info = client_info | self.get_config()
                client_info["stats"][split_name] = split_stats
                client_info["size_mb"][split_name] = (
                    os.path.getsize(file_path) / 1024 / 1024
                )

            # check if client_info["samples"] has (0, x, x) or not, if yes, skip this client
            if any(
                int(client_info["samples"][split_name]["x"].split(",")[0][1:]) == 0
                for split_name in client_info["samples"]
            ):
                print(
                    f"Skipping client {client_id} due to zero samples in one of the splits."
                )
                continue
            else:
                self.info.append(client_info)
                print(json.dumps(self.info[-1], indent=4))
                print(f"Time elapsed: {time.time() - start_time:.2f} seconds\n{'='*50}")

    def download(self):
        pass

    def execute(self):
        if not os.path.exists(self.path_raw) or len(os.listdir(self.path_raw)) == 0:
            self.download()
        self.dir()
        self.fix_params()
        if self.check():
            return
        self.get_file_paths()
        self.generate()
        self.save_info()


class CustomDataset(BaseDataset):
    def execute(self):
        self.dir()
        self.generate()
        self.save_info()

    def dir(self):
        name = f"seq_{self.seq_len}-offset_{self.offset_len}-pred_{self.pred_len}"
        self.path_save = os.path.join(self.save_path, name)
        self.path_info = os.path.join(self.path_save, "info.json")
        os.makedirs(self.path_save, exist_ok=True)

    def generate(self):
        for info in self.sets:
            for k, v in info.items():
                if k not in ["dataset", "column_train", "column_target"]:
                    setattr(self.configs, k, v)
            dataset = info["dataset"](configs=self.configs)
            for k, v in info.items():
                if k in ["column_train", "column_target"]:
                    setattr(dataset, k, v)
            dataset.execute()
            self.info.extend(dataset.info)
        for idx, info in enumerate(self.info):
            info["client"] = idx


class CustomOnSingleDataset(CustomDataset):
    def dir(self):
        self.path_save = self.save_path
        self.path_info = os.path.join(self.path_save, "info.json")
        os.makedirs(self.path_save, exist_ok=True)

    def generate(self):
        """
        Processes multiple times, automatically calculates slice sizes, and extracts info.
        WARNING: Leads to incorrect .npz files due to overwriting.
        """
        print("--- Starting Generation Loop (Auto Slice Method) ---")
        combined_info_list = []
        start_index_for_next_slice = 0
        total_clients_found = None  # Will be determined after first run

        # Store original config value if it exists, to restore later
        original_pred_len = getattr(self.configs, "output_len", None)

        for i, config_info in enumerate(self.sets):
            current_output_len = config_info["output_len"]
            print(
                f"\nProcessing config {i+1}/{len(self.sets)}: output_len = {current_output_len}"
            )

            # 1. Update shared configs object
            setattr(self.configs, "output_len", current_output_len)

            # 2. Create and execute dataset instance
            print(f"  Instantiating and executing {config_info['dataset'].__name__}...")
            dataset_instance = config_info["dataset"](configs=self.configs)
            # Apply column overrides if any
            for k, v in config_info.items():
                if k in ["column_train", "column_target"]:
                    setattr(dataset_instance, k, v)
            dataset_instance.execute()  # This reruns processing and potentially overwrites files

            if not hasattr(dataset_instance, "info") or not dataset_instance.info:
                print(f"  Warning: dataset_instance.info is empty. Skipping slice.")
                continue

            # --- Automatic Slice Calculation ---
            len_info = len(dataset_instance.info)  # Total clients found in this run
            if total_clients_found is None:
                total_clients_found = len_info  # Set the total based on the first run
                print(f"  Determined total clients found: {total_clients_found}")
            elif total_clients_found != len_info:
                # This case indicates inconsistency, which is problematic for slicing
                print(
                    f"  Warning: Number of clients found ({len_info}) differs from first run ({total_clients_found}). Slicing might be incorrect."
                )
                # Optionally, update total_clients_found or raise an error depending on desired behavior
                # total_clients_found = len_info # Update if we trust the latest run more

            len_config = len(self.sets)
            base_count = total_clients_found // len_config
            remainder = total_clients_found % len_config

            # Determine the count for *this specific* configuration index `i`
            if i < remainder:
                count_for_this_config = base_count + 1
            else:
                count_for_this_config = base_count
            # --------------------------------

            # --- Slicing Logic ---
            end_index_for_this_slice = (
                start_index_for_next_slice + count_for_this_config
            )
            print(f"  Auto-calculated count for config {i+1}: {count_for_this_config}")
            print(
                f"  Slicing dataset_instance.info [{start_index_for_next_slice}:{end_index_for_this_slice}]"
            )

            # Slice the info list generated *in this iteration*
            actual_end_index = min(end_index_for_this_slice, len_info)  # Boundary check
            actual_start_index = min(start_index_for_next_slice, actual_end_index)

            if actual_start_index >= actual_end_index:
                print(
                    f"  Warning: Slice [{actual_start_index}:{actual_end_index}] is empty."
                )
                client_info_slice = []
            else:
                client_info_slice = dataset_instance.info[
                    actual_start_index:actual_end_index
                ]
                print(f"  Sliced {len(client_info_slice)} entries.")

            combined_info_list.extend(client_info_slice)

            # Update start index for the *next* iteration
            start_index_for_next_slice = end_index_for_this_slice
            # --------------------

            print(
                f"  Finished loop for output_len={current_output_len}. Total info: {len(combined_info_list)}"
            )

        # Restore original config value
        if original_pred_len is not None:
            print(f"\nRestoring self.configs.output_len to {original_pred_len}")
            setattr(self.configs, "output_len", original_pred_len)
        elif hasattr(self.configs, "output_len"):
            delattr(self.configs, "output_len")

        # Assign final list and re-index
        self.info = combined_info_list
        print("\nRe-indexing final client list...")
        for idx, client_data in enumerate(self.info):
            client_data["client"] = idx
        print(f"Final self.info list contains {len(self.info)} entries.")
