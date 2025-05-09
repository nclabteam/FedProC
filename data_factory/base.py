import io
import json
import os
import re
import time
import zipfile

import gdown
import numpy as np
import polars as pl
import requests
from statsmodels.tsa.seasonal import STL


class BaseDataset:
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
    ):
        """
        Initializes the BaseDataset instance.

        Args:
            configs (dict): Configuration dictionary containing necessary parameters
                            like input/output lengths, paths, column names, etc.
        """
        self.configs = configs
        self.seq_len = configs.input_len
        self.pred_len = configs.output_len
        self.offset_len = configs.offset_len
        self.train_ratio = configs.train_ratio
        self.info = []

        self.path_raw = None
        self.save_path = None
        self.column_date = None
        self.column_target = None
        self.column_train = None
        self.granularity = None
        self.granularity_unit = None

    @staticmethod
    def convert_granularity_unit(granularity_unit: str) -> str:
        """
        Converts human-readable time granularity units to Polars/Pandas frequency strings.

        Args:
            granularity_unit (str): The full name of the time unit (e.g., "hour", "day").

        Returns:
            str: The corresponding frequency string abbreviation (e.g., "h", "d"),
                 or the original string if no match is found.
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
        Extracts time-based features from a datetime series based on the frequency.

        Features are normalized to the range [-0.5, 0.5]. The selection of features
        depends on the provided frequency to avoid redundant or constant features
        (e.g., 'second_of_minute' is irrelevant for daily data).

        Args:
            dates (pl.Series): A Polars Series containing datetime objects.
            freq (str): The frequency string (e.g., 'h', 'd', 'm') indicating the
                        granularity of the time series.

        Returns:
            pl.DataFrame: A Polars DataFrame with extracted time features as columns.

        Raises:
            ValueError: If the provided frequency `freq` is not supported.
        """

        # Define all possible time features and their normalization functions
        features = {
            "second_of_minute": lambda dt: dt.dt.second() / 59.0 - 0.5,
            "minute_of_hour": lambda dt: dt.dt.minute() / 59.0 - 0.5,
            "hour_of_day": lambda dt: dt.dt.hour() / 23.0 - 0.5,
            "day_of_week": lambda dt: dt.dt.weekday() / 6.0 - 0.5,
            "day_of_month": lambda dt: (dt.dt.day() - 1) / 30.0 - 0.5,
            "day_of_year": lambda dt: (dt.dt.ordinal_day() - 1) / 365.0 - 0.5,
            "month_of_year": lambda dt: (dt.dt.month() - 1) / 11.0 - 0.5,
            "week_of_year": lambda dt: (dt.dt.week() - 1) / 52.0 - 0.5,
        }

        # Map frequencies to the relevant time features
        freq_mapping = {
            "y": [],
            "q": ["month_of_year"],
            "mo": ["month_of_year"],
            "w": ["day_of_month", "week_of_year"],
            "d": ["day_of_week", "day_of_month", "day_of_year"],
            "h": ["hour_of_day", "day_of_week", "day_of_month", "day_of_year"],
            "m": [
                "minute_of_hour",
                "hour_of_day",
                "day_of_week",
                "day_of_month",
                "day_of_year",
            ],
            "s": [
                "second_of_minute",
                "minute_of_hour",
                "hour_of_day",
                "day_of_week",
                "day_of_month",
                "day_of_year",
            ],
            "ms": [
                "second_of_minute",
                "minute_of_hour",
                "hour_of_day",
                "day_of_week",
                "day_of_month",
                "day_of_year",
            ],
            "us": [
                "second_of_minute",
                "minute_of_hour",
                "hour_of_day",
                "day_of_week",
                "day_of_month",
                "day_of_year",
            ],
            "ns": [
                "second_of_minute",
                "minute_of_hour",
                "hour_of_day",
                "day_of_week",
                "day_of_month",
                "day_of_year",
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
    def subdf_from_indices(df: pl.DataFrame, date_column: str, start, end):
        """
        Filters a DataFrame to select rows within a specified date range.

        The range is inclusive of the start date and exclusive of the end date ([start, end)).

        Args:
            df (pl.DataFrame): The input DataFrame containing a date column.
            date_column (str): The name of the column containing date/datetime objects.
            start (pl.datatypes.Date or datetime): The starting date/time (inclusive).
            end (pl.datatypes.Date or datetime): The ending date/time (exclusive).

        Returns:
            pl.DataFrame: A filtered DataFrame containing rows within the specified date range.
        """
        return df.filter((pl.col(date_column) >= start) & (pl.col(date_column) < end))

    @staticmethod
    def _memory_unit_conversion(before: float, after: float):
        """
        Converts memory sizes (in bytes) to a more readable unit (KB, MB, GB, etc.).

        Args:
            before (float): The initial memory size in bytes.
            after (float): The final memory size in bytes.

        Returns:
            tuple[float, float, str]: A tuple containing the converted 'before' size,
                                      converted 'after' size, and the unit string.
        """
        units = ["bytes", "KB", "MB", "GB", "TB", "PB"]
        unit_index = 0

        # Iteratively divide by 1024 to find the appropriate unit
        while before >= 1024 and unit_index < len(units) - 1:
            before /= 1024
            after /= 1024
            unit_index += 1

        return before, after, units[unit_index]

    def _print_info(self, before: float, after: float, data_type: str):
        """
        Prints memory usage information before and after an optimization process.

        Calculates the percentage reduction and formats the output using appropriate units.

        Args:
            before (float): Memory usage before optimization (in bytes).
            after (float): Memory usage after optimization (in bytes).
            data_type (str): A string describing the type of data being optimized
                             (e.g., "Polars DataFrame").
        """
        reduction = 100.0 * (before - after) / before
        reduction_str = f"{reduction:.2f}% reduction"
        before, after, unit = self._memory_unit_conversion(before, after)

        print(
            f"{data_type} memory usage: {before:.4f} {unit} ⇒ {after:.4f} {unit} ({reduction_str})"
        )

    @staticmethod
    def _cast_to_optimal_type(c_min: float, c_max: float, current_type: str):
        """
        Finds the smallest NumPy numeric type that can represent a range [c_min, c_max].

        Checks integer types if `current_type` is 'int', otherwise checks float types.

        Args:
            c_min (float): The minimum value in the column.
            c_max (float): The maximum value in the column.
            current_type (str): Either 'int' or 'float', indicating the original type family.

        Returns:
            np.dtype | None: The smallest NumPy dtype that fits the range, or None if no
                             suitable type is found within the predefined lists.
        """
        numeric_int_types = [np.int8, np.int16, np.int32, np.int64]
        numeric_float_types = [np.float16, np.float32, np.float64]

        # Find the smallest numeric type that fits the given range.
        dtype_list = numeric_int_types if current_type == "int" else numeric_float_types
        for dtype in dtype_list:
            if np.can_cast(c_min, dtype) and np.can_cast(c_max, dtype):
                return dtype
        return None

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

    @staticmethod
    def download_and_extract(url: str, save_path: str):
        """
        Downloads a ZIP file from a URL and extracts its contents.

        Args:
            url (str): The URL of the ZIP file.
            save_path (str): The directory where the contents should be extracted.

        Raises:
            requests.exceptions.RequestException: If the download fails.
            zipfile.BadZipFile: If the downloaded file is not a valid ZIP archive.
        """
        response = requests.get(url)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(save_path)

    @staticmethod
    def download_file(url: str, save_path: str):
        """
        Downloads a file from a URL and saves it to a specified path.

        Uses streaming to handle potentially large files efficiently.

        Args:
            url (str): The URL of the file to download.
            save_path (str): The local path (including filename) where the file should be saved.
        """
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
        else:
            print(f"Failed to download {url} (Status Code: {response.status_code})")

    @staticmethod
    def download_from_google_drive(file_id: str, save_path: str):
        """
        Downloads a file from Google Drive using its file ID.

        Requires the `gdown` library to be installed.

        Args:
            file_id (str): The Google Drive file ID.
            save_path (str): The local path (including filename) to save the file.
        """
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, save_path, quiet=False)

    @staticmethod
    def split_column_into_files(
        df: pl.DataFrame,
        path: str,
        station_column: str,
        date_column: str | None = None,
        remove_station_column: bool = True,
    ):
        """
        Splits a DataFrame into multiple CSV files based on unique values in a specific column.

        Each unique value in `station_column` results in a separate CSV file named
        after that value.

        Args:
            df (pl.DataFrame): The input DataFrame.
            path (str): The directory where the output CSV files will be saved.
            station_column (str): The name of the column whose unique values define the splits.
            date_column (str, optional): If provided, sorts each subset by this date column
                                         before saving. Defaults to None.
            remove_station_column (bool, optional): If True, removes the `station_column`
                                                    from the output files. Defaults to True.
        """
        for station in df[station_column].unique().to_list():
            sdf = df.filter(df[station_column] == station)
            sdf = sdf.sort(date_column) if date_column else sdf
            sdf = sdf.drop(station_column) if remove_station_column else sdf
            sdf.write_csv(os.path.join(path, f"{station}.csv"))

    @staticmethod
    def split_columns_into_files(
        df: pl.DataFrame,
        path: str,
        date_column: str,
        new_column_name: str = "Value",
        keep_old_column_name_as_filename: bool = False,
    ):
        """
        Splits a DataFrame into multiple CSV files, one for each column (excluding the date column).

        Each output file contains the date column and one other column, renamed to `new_column_name`.
        Files are named either sequentially (0.csv, 1.csv, ...) or using the original column names.

        Args:
            df (pl.DataFrame): The input DataFrame.
            path (str): The directory where the output CSV files will be saved.
            date_column (str): The name of the date/time column to include in each file.
            new_column_name (str, optional): The name to give the value column in the output files.
                                             Defaults to "Value".
            keep_old_column_name_as_filename (bool, optional): If True, uses original column names
                                                              for filenames. Otherwise, uses sequential integers.
                                                              Defaults to False.
        """
        cols = df.columns
        cols.remove(date_column)
        for idx, col in enumerate(cols):
            if col == date_column:
                continue
            sdf = df.select([date_column, col])
            sdf = sdf.rename({col: new_column_name})
            if keep_old_column_name_as_filename:
                sdf.write_csv(os.path.join(path, f"{col}.csv"))
            else:
                sdf.write_csv(os.path.join(path, f"{idx}.csv"))

    @staticmethod
    def read(path: str) -> pl.DataFrame | None:
        """
        Reads a CSV file into a Polars DataFrame with automatic date parsing.

        Handles potential `NoDataError` if the file is empty.

        Args:
            path (str): The path to the CSV file.

        Returns:
            pl.DataFrame | None: The loaded DataFrame, or None if the file is empty.
        """
        try:
            df = pl.read_csv(path, try_parse_dates=True)
            return df
        except pl.exceptions.NoDataError:
            print(f"Empty file: {path}")
            return None

    @staticmethod
    def get_transition_value(df: pl.DataFrame):
        """
        Calculates a 'transition value' for each numerical column in the DataFrame.

        This metric is based on the algorithm described in:
        Fulcher, B. D., & Jones, N. S. (2014). Highly comparative feature-based
        time-series classification. IEEE Transactions on Knowledge and Data
        Engineering, 26(12), 3026-3037.
        (Specifically, the 'TransitionMeasure' feature CO_AutoPersist_z_tau_3)

        Args:
            df (pl.DataFrame): Input DataFrame with numerical time series columns.
                               Assumes columns are numerical.

        Returns:
            pl.DataFrame: A DataFrame with one row, where the first column is 'variable'
                          (value 'transition') and subsequent columns contain the calculated
                          transition value (delta) for each corresponding column in the input df.
        """

        def first_zero_ac(X: np.ndarray):
            """Computes the lag of the first zero crossing of the ACF."""
            n = len(X)

            if n < 2:
                return 1  # Handle very short series

            X_centered = X - np.mean(X)

            # Compute autocorrelation using numpy's correlate
            acf = np.correlate(X_centered, X_centered, mode="full")[n - 1 :]

            # Normalize ACF (optional, but standard)
            variance = np.var(X)
            if variance == 0:
                return 1  # Handle constant series
            acf /= variance * n

            # Find the first lag where ACF drops below zero
            zero_crossings = np.where(acf < 0)[0]
            if len(zero_crossings) > 0:
                return zero_crossings[0]  # Return the first lag index
            else:
                return n  # If ACF never crosses zero, return series length

        # Loop through each column in the DataFrame to compute the transition value
        transition_data = {"variable": "transition"}

        for column in df.columns:
            # Step 1: Calculate the first zero crossing of the autocorrelation function
            X = df[column].to_numpy()
            tau = first_zero_ac(X)

            # Step 2: Downsample the time series with stride tau
            Y = X[::tau]

            # Step 3: Sort the downsampled array and create the index I
            I = np.argsort(Y)

            # Step 4: Characterize Y to obtain Z
            Z = np.floor(I / (len(Y) / 3)).astype(int)  # Map to 3 values: 0, 1, 2

            # Step 5: Generate the transition matrix M
            M = np.zeros((3, 3), dtype=int)
            for j in range(len(Y) - 1):
                M[Z[j], Z[j + 1]] += 1

            # Step 6: Normalize the matrix M to get M'
            M_prime = M / len(Y)

            # Step 7: Compute the covariance matrix C
            C = np.cov(M_prime)

            # Step 8: Calculate the trace of the covariance matrix
            delta = np.trace(C)

            transition_data[column] = delta

        return pl.DataFrame(transition_data)

    @staticmethod
    def calculate_shifting_values(df: pl.DataFrame, m: int = 10):
        # Initialize a dictionary to store shifting values for each column
        delta_data = {"variable": f"shifting({m=})"}

        # Loop through all columns in the DataFrame
        for column in df.columns:
            # Step 1: Normalize the time series by calculating the z-score
            X = df[column].to_numpy()
            Z = (X - np.mean(X)) / np.std(X)

            # Step 2: Calculate min and max of Z
            Z_min, Z_max = np.min(Z), np.max(Z)

            # Step 3: Define the set S of thresholds
            thresholds = [
                Z_min + (i - 1) * (Z_max - Z_min) / m for i in range(1, m + 1)
            ]

            M = []
            # Step 4: For each threshold, calculate the set K and the median M_i
            for s in thresholds:
                K = np.where(Z > s)[0]  # Get the indices where Z > s
                if len(K) > 0:
                    M_i = np.median(K)
                else:
                    M_i = None
                M.append(M_i)

            # Step 5: Filter out None values from M
            M = [m for m in M if m is not None]

            # Step 6: Min-Max normalize M (after filtering None values)
            if len(M) > 0:  # Ensure there's at least one value in M
                M_min, M_max = np.min(M), np.max(M)
                M_prime = (M - M_min) / (M_max - M_min) if M_max > M_min else M
                # Step 7: Return the median of M_prime as the shifting value δ
                delta = np.median(M_prime)
            else:
                delta = None  # In case there were no valid values in M

            delta_data[column] = delta  # Store the shifting value for the column

        return pl.DataFrame(delta_data)

    @staticmethod
    def compute_trend_seasonal_strength_auto(
        df: pl.DataFrame,
        granularity: int,
        granularity_unit: str,
        max_nan_fraction: float = 0.1,
    ):
        """
        Computes trend and seasonal strength for all numerical columns using auto-selected seasonal period.

        Parameters:
            df (pl.DataFrame): Time series dataset.
            granularity (int): Time step size (e.g., 1 for hourly, 1 for daily).
            granularity_unit (str): Unit of time granularity ("minute", "hour", "day", etc.).

        Returns:
            pl.DataFrame: A DataFrame with trend & seasonal strengths for each column.
        """
        seasonal_period = {
            "m": max(60 // granularity, 1),
            "h": max(24 // granularity, 1),
            "d": max(7 // granularity, 1),
            "w": max(52 // granularity, 1),
            "mo": max(12 // granularity, 1),
        }
        trend_results = {"variable": f"trend_strength"}
        seasonal_results = {"variable": f"seasonal_strength"}

        for col in df.columns:
            ts_series = df[col]
            original_length = len(ts_series)

            # Check for NaN values and handle them
            nan_count = ts_series.is_null().sum()
            nan_fraction = nan_count / original_length

            if 0 < nan_fraction < max_nan_fraction:
                ts_series = ts_series.interpolate()
            elif nan_fraction >= max_nan_fraction:
                trend_results[col] = 0.0
                seasonal_results[col] = 0.0
                continue

            ts_values = ts_series.to_numpy()
            stl = STL(ts_values, period=seasonal_period[granularity_unit], robust=True)
            result = stl.fit()

            trend = result.trend
            seasonal = result.seasonal
            residual = result.resid

            var_residual = np.var(residual)
            var_diff_seasonal = np.var(ts_values - seasonal)
            var_diff_trend = np.var(ts_values - trend)

            trend_strength = max(0.0, 1 - (var_residual / var_diff_seasonal))
            seasonal_strength = max(0.0, 1 - (var_residual / var_diff_trend))

            trend_results[col] = trend_strength
            seasonal_results[col] = seasonal_strength

        return pl.DataFrame([trend_results, seasonal_results])

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
                self.compute_trend_seasonal_strength_auto(
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
        if self.total_null != 0:
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

    def reduce_polars_df(self, df, info=False):
        before = df.estimated_size()

        np_to_pl_type_mapping = {
            np.int8: pl.Int8,
            np.int16: pl.Int16,
            np.int32: pl.Int32,
            np.int64: pl.Int64,
            np.uint8: pl.UInt8,
            np.uint16: pl.UInt16,
            np.uint32: pl.UInt32,
            np.uint64: pl.UInt64,
            np.float16: pl.Float32,
            np.float32: pl.Float32,
            np.float64: pl.Float64,
        }

        for col in df.columns:
            col_type = df[col].dtype

            if col_type == pl.Utf8:
                df = df.with_columns(df[col].cast(pl.Categorical))
            elif col_type == pl.Datetime:
                # Check if all timestamps are at midnight (no time component)
                if (
                    df[col].dt.hour().sum() == 0
                    and df[col].dt.minute().sum() == 0
                    and df[col].dt.second().sum() == 0
                ):
                    df = df.with_columns(df[col].cast(pl.Date))
            elif col_type in np_to_pl_type_mapping.values():
                c_min, c_max = df[col].min(), df[col].max()
                optimal_type = self._cast_to_optimal_type(
                    c_min, c_max, "float" if "float" in str(col_type) else "int"
                )
                if optimal_type:
                    polars_type = np_to_pl_type_mapping.get(optimal_type)
                    if polars_type:
                        df = df.with_columns(df[col].cast(polars_type))

        if info:
            self._print_info(before, df.estimated_size(), "Polars DataFrame")
        return df

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
                # split_time_df = self.extract_time_features(dates=split_df[self.column_date], freq=self.granularity_unit)
                # null_mask = split_df.with_columns(
                #     pl.concat_list(pl.all().is_null()).alias("null_mask")
                # )["null_mask"].list.any()
                # split_time_df = split_time_df.with_columns(
                #     [
                #         pl.when(null_mask)
                #         .then(None)
                #         .otherwise(split_time_df[col])
                #         .alias(col)
                #         for col in split_time_df.columns
                #     ]
                # )

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
                # split_x_mark, split_y_mark = self.split_x_y(
                #     df=split_time_df,
                #     x_used_cols=split_time_df.columns,
                #     y_used_cols=split_time_df.columns,
                # )

                # Save split data
                file_path = os.path.join(
                    getattr(self, f"path_{split_name}"), f"{client_id}.npz"
                )
                np.savez_compressed(
                    file=file_path,
                    # x_mark=split_x_mark,
                    # y_mark=split_y_mark,
                    x=split_x,
                    y=split_y,
                )
                client_info["paths"][split_name] = file_path
                client_info = client_info | self.get_config()
                client_info["stats"][split_name] = split_stats
                client_info["size_mb"][split_name] = (
                    os.path.getsize(file_path) / 1024 / 1024
                )

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
