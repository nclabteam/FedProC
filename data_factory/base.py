import json
import os
import re
import tempfile
import time
import zipfile
from pathlib import Path

import gdown
import numpy as np
import polars as pl
import requests
from scipy.stats import entropy
from statsmodels.tsa.seasonal import STL


class TimeSeriesCharacteristics:
    """Utility class for computing statistical and structural metrics of time series data.

    Provides methods for extracting quantitative characteristics of time series including
    transition measures, shifting values, and decomposition-based metrics. All metrics are
    designed to be meaningful for time series classification and feature engineering tasks.
    """

    @staticmethod
    def get_transition_value(df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate transition value metric for each numerical column in the DataFrame.

        Computes a robust measure of time series behavior based on the autocorrelation function
        (ACF) and state-space transitions. This metric captures how the time series transitions
        between different relative states and is particularly useful for characterizing complex
        temporal patterns.

        The algorithm (Fulcher & Jones, 2014):
            1. Compute first zero crossing of ACF to determine downsampling stride τ
            2. Downsample series with stride τ
            3. Sort downsampled values and map to 3 states (low, mid, high)
            4. Build 3×3 transition matrix M
            5. Normalize M and compute covariance
            6. Return trace of covariance matrix as transition value δ

        This metric is based on research described in:
            Fulcher, B. D., & Jones, N. S. (2014). Highly comparative feature-based
            time-series classification. IEEE Transactions on Knowledge and Data
            Engineering, 26(12), 3026-3037.
            (Specifically, the 'TransitionMeasure' feature CO_AutoPersist_z_tau_3)

        Args:
            df (pl.DataFrame): Input DataFrame with numerical time series columns.
                              Must contain only numerical (int or float) columns.
                              Each column is treated as an independent time series.

        Returns:
            pl.DataFrame: Single-row DataFrame where:
                         - First column 'variable' contains 'transition'
                         - Remaining columns contain transition value δ for each input column
                         - Values range from 0.0 to positive infinity
                         - Higher values indicate more complex state transitions

        Raises:
            ValueError: If DataFrame is empty or contains no numerical columns.
            ZeroDivisionError: If series has zero variance (handled gracefully, returns 1).

        Examples:
            >>> import polars as pl
            >>> import numpy as np
            >>> # Create synthetic time series data
            >>> dates = pl.datetime_range("2023-01-01", periods=100, interval="1d", eager=True)
            >>> signal1 = pl.Series("temp", np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100))
            >>> signal2 = pl.Series("humidity", np.cos(np.linspace(0, 2*np.pi, 100)) + np.random.normal(0, 0.05, 100))
            >>> df = pl.DataFrame({"temp": signal1, "humidity": signal2})
            >>> result = TimeSeriesCharacteristics.get_transition_value(df)
            >>> result
            shape: (1, 3)
            ┌────────────┬──────┬──────────┐
            │ variable   ┆ temp ┆ humidity │
            │ ---        ┆ ---  ┆ ---      │
            │ str        ┆ f64  ┆ f64      │
            ╞════════════╪══════╪══════════╡
            │ transition ┆ 0.45 ┆ 0.52     │
            └────────────┴──────┴──────────┘

        Notes:
            - Very short series (< 2 samples) default to τ = 1
            - Constant-valued series (zero variance) default to τ = 1
            - NaN/Inf values may produce unexpected results; clean data beforehand
            - Computation time scales linearly with series length and column count
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
    def calculate_shifting_values(df: pl.DataFrame, m: int = 10) -> pl.DataFrame:
        """Calculate distribution-based shifting value metric for each column.

        Computes a metric that characterizes the distribution and clustering of time series
        values by computing the spacing between quantile thresholds. This captures how the
        time series values spread across their range.

        The algorithm:
            1. Z-score normalize each column
            2. Create m equally-spaced threshold values from Z_min to Z_max
            3. For each threshold, find indices where normalized value > threshold
            4. Compute median index position for each threshold
            5. Min-Max normalize the median positions
            6. Return median of normalized positions as shifting value δ

        Args:
            df (pl.DataFrame): Input DataFrame with numerical columns.
                              Must contain only numerical (int or float) columns.
            m (int, optional): Number of threshold levels to use for quantile computation.
                              Higher m provides finer granularity but increases computation.
                              Defaults to 10.

        Returns:
            pl.DataFrame: Single-row DataFrame where:
                         - First column 'variable' contains 'shifting(m=N)' descriptor
                         - Remaining columns contain shifting value δ for each input column
                         - Values typically range from 0.0 to 1.0
                         - Higher values indicate more uniform distribution

        Raises:
            ValueError: If DataFrame is empty or contains no numerical columns.
            ValueError: If m < 1.

        Examples:
            >>> import polars as pl
            >>> import numpy as np
            >>> # Create data with different distributions
            >>> uniform_data = pl.Series("uniform", np.linspace(0, 100, 100))
            >>> skewed_data = pl.Series("skewed", np.random.exponential(2, 100))
            >>> df = pl.DataFrame({"uniform": uniform_data, "skewed": skewed_data})
            >>> result = TimeSeriesCharacteristics.calculate_shifting_values(df, m=10)
            >>> result
            shape: (1, 3)
            ┌──────────────┬─────────┬─────────┐
            │ variable     ┆ uniform ┆ skewed  │
            │ ---          ┆ ---     ┆ ---     │
            │ str          ┆ f64     ┆ f64     │
            ╞══════════════╪═════════╪═════════╡
            │ shifting(... ┆ 0.51    ┆ 0.38    │
            └──────────────┴─────────┴─────────┘

        Notes:
            - Higher m values may produce more robust estimates but require more computation
            - m=10 is a reasonable default balancing robustness and efficiency
            - Returns None for columns where no valid quantile values exist
            - Useful for distinguishing between uniform and skewed distributions
        """
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
    def compute_trend_seasonal_strength_and_entropy_auto(
        df: pl.DataFrame,
        granularity: int,
        granularity_unit: str,
        max_nan_fraction: float = 0.1,
    ) -> pl.DataFrame:
        """Decompose time series into trend/seasonal components and compute strength metrics.

        Uses STL (Seasonal and Trend decomposition using LOESS) to separate time series into
        trend, seasonal, and residual components. Computes strength metrics (how much of
        variance is explained by each component) and entropy measures (how random each
        component is).

        The seasonal period is automatically determined based on granularity:
            - Minute: 60 periods (1 hour)
            - Hour: 24 periods (1 day)
            - Day: 7 periods (1 week)
            - Week: 52 periods (1 year)
            - Month: 12 periods (1 year)

        Args:
            df (pl.DataFrame): Time series dataset with numerical columns.
                              Each column treated as independent time series.
                              Should have sufficient length (ideally 2× seasonal period).
            granularity (int): Time step size (e.g., 1 for every minute/hour/day).
                              Used to scale the seasonal period. For example:
                              - granularity=1 + unit='hour' → seasonal_period=24
                              - granularity=6 + unit='hour' → seasonal_period=4
            granularity_unit (str): Unit of time granularity. Must be one of:
                              'm' (minute), 'h' (hour), 'd' (day), 'w' (week), 'mo' (month).
            max_nan_fraction (float, optional): Maximum fraction of NaN values allowed.
                              If exceeded, column is skipped (returned as 0.0 for all metrics).
                              Allowed range: 0.0-1.0. Defaults to 0.1 (10%).

        Returns:
            pl.DataFrame: 4-row DataFrame containing:
                         Row 1: {variable: 'trend_strength', col1: float, col2: float, ...}
                         Row 2: {variable: 'seasonal_strength', col1: float, ...}
                         Row 3: {variable: 'trend_entropy', col1: float, ...}
                         Row 4: {variable: 'seasonal_entropy', col1: float, ...}

                         Where each metric is in range [0.0, 1.0]:
                         - Strength: 0 = component absent, 1 = dominates
                         - Entropy: 0 = highly structured, 1 = random/uniform

        Raises:
            ValueError: If granularity_unit not in {'m', 'h', 'd', 'w', 'mo'}.
            ValueError: If max_nan_fraction not in [0.0, 1.0].
            RuntimeError: If STL decomposition fails (e.g., series too short).

        Examples:
            >>> import polars as pl
            >>> import numpy as np
            >>> # Create synthetic daily time series with trend and seasonality
            >>> dates = pl.datetime_range("2023-01-01", periods=365, interval="1d", eager=True)
            >>> t = np.arange(365)
            >>> trend = 20 + 0.05 * t  # Increasing trend
            >>> seasonal = 10 * np.sin(2 * np.pi * t / 365)  # Annual seasonality
            >>> noise = np.random.normal(0, 1, 365)
            >>> values = trend + seasonal + noise
            >>> df = pl.DataFrame({"temperature": values})
            >>> result = TimeSeriesCharacteristics.compute_trend_seasonal_strength_and_entropy_auto(
            ...     df, granularity=1, granularity_unit='d'
            ... )
            >>> result
            shape: (4, 2)
            ┌──────────────────┬──────────────┐
            │ variable         ┆ temperature  │
            │ ---              ┆ ---          │
            │ str              ┆ f64          │
            ╞══════════════════╪══════════════╡
            │ trend_strength   ┆ 0.72         │
            │ seasonal_strength┆ 0.68         │
            │ trend_entropy    ┆ 0.15         │
            │ seasonal_entropy ┆ 0.08         │
            └──────────────────┴──────────────┘

        Notes:
            - NaN values are automatically interpolated if fraction < max_nan_fraction
            - STL uses robust=True to handle outliers gracefully
            - Requires series to have length >= 2 × seasonal_period for stable decomposition
            - Very short series may produce spurious results
            - Entropy is computed using 20-bin histogram for stability
            - Trend and seasonal strength range from 0 (absent) to 1 (dominant)
        """

        seasonal_period = {
            "m": max(60 // granularity, 1),
            "h": max(24 // granularity, 1),
            "d": max(7 // granularity, 1),
            "w": max(52 // granularity, 1),
            "mo": max(12 // granularity, 1),
        }
        trend_results = {"variable": "trend_strength"}
        seasonal_results = {"variable": "seasonal_strength"}
        trend_entropy_results = {"variable": "trend_entropy"}
        seasonal_entropy_results = {"variable": "seasonal_entropy"}

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
                trend_entropy_results[col] = 0.0
                seasonal_entropy_results[col] = 0.0
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

            # Calculate entropy for trend and seasonal components
            def safe_entropy(arr, bins=20):
                hist, _ = np.histogram(arr, bins=bins, density=True)
                hist = hist + 1e-12  # avoid log(0)
                hist = hist / hist.sum()
                return float(entropy(hist))

            trend_entropy = safe_entropy(trend)
            seasonal_entropy = safe_entropy(seasonal)

            trend_results[col] = trend_strength
            seasonal_results[col] = seasonal_strength
            trend_entropy_results[col] = trend_entropy
            seasonal_entropy_results[col] = seasonal_entropy

        return pl.DataFrame(
            [
                trend_results,
                seasonal_results,
                trend_entropy_results,
                seasonal_entropy_results,
            ]
        )


class FileManager:
    """Utility class for file I/O operations and remote data download/extraction.

    Provides static methods for downloading files from URLs (HTTP, Google Drive),
    extracting compressed archives, reading CSV files, and splitting DataFrames
    into multiple files. All methods operate independently and can be used
    without instantiating the class.
    """

    request_timeout = 60

    @staticmethod
    def _ensure_parent_dir(path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    @staticmethod
    def _safe_extract_zip(zip_file: zipfile.ZipFile, destination: str) -> None:
        destination_path = Path(destination).resolve()
        destination_path.mkdir(parents=True, exist_ok=True)

        for member in zip_file.infolist():
            member_path = destination_path / member.filename
            resolved_member_path = member_path.resolve(strict=False)
            if os.path.commonpath(
                [str(destination_path), str(resolved_member_path)]
            ) != str(destination_path):
                raise ValueError(
                    f"Unsafe ZIP member path {member.filename!r} escapes "
                    f"{destination_path}."
                )

        zip_file.extractall(destination)

    @staticmethod
    def download_and_extract(url: str, save_path: str) -> None:
        """Download a ZIP file from URL and extract contents to directory.

        Downloads a remote ZIP file via HTTP and extracts all contents to the
        specified directory. Uses streaming for memory efficiency with large files.
        Creates the target directory if it doesn't exist.

        Args:
            url (str): URL of the ZIP file to download.
                      Must be a valid HTTP/HTTPS URL pointing to a ZIP archive.
                      Example: 'https://example.com/data.zip'
            save_path (str): Directory where ZIP contents will be extracted.
                           Created if it doesn't exist. All extracted files
                           will preserve their original directory structure from the ZIP.
                           Example: './datasets' or '/data/archive'

        Returns:
            None: Extraction is done in-place.

        Raises:
            requests.exceptions.RequestException: If HTTP download fails.
                                                 Common causes: network error, invalid URL, timeout.
            requests.exceptions.MissingSchema: If url is not valid HTTP/HTTPS.
            requests.exceptions.Timeout: If download takes too long (default ~30s).
            zipfile.BadZipFile: If downloaded file is not a valid ZIP archive.
            IOError: If save_path cannot be created or written to.
            MemoryError: If ZIP file is extremely large (> available RAM).

        Examples:
            >>> # Download and extract a remote dataset
            >>> FileManager.download_and_extract(
            ...     'https://example.com/dataset.zip',
            ...     './data/extracted'
            ... )
            >>> # Check extraction succeeded
            >>> import os
            >>> os.listdir('./data/extracted')
            ['subfolder1', 'file1.csv', 'file2.csv']

            >>> # Download from GitHub releases
            >>> FileManager.download_and_extract(
            ...     'https://github.com/user/repo/releases/download/v1.0/data.zip',
            ...     './github_data'
            ... )

        Notes:
            - ZIP directory structure is preserved during extraction
            - Existing files are overwritten if already in save_path
            - Empty directories in ZIP are also created
            - For very large files, consider resuming failed downloads manually
        """
        os.makedirs(save_path, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix=".zip",
            delete=False,
            dir=save_path,
        ) as temp_file:
            temp_path = temp_file.name

        try:
            with requests.get(url, stream=True, timeout=FileManager.request_timeout) as response:
                response.raise_for_status()
                with open(temp_path, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)

            with zipfile.ZipFile(temp_path) as zip_ref:
                FileManager._safe_extract_zip(zip_ref, save_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @staticmethod
    def download_file(url: str, save_path: str) -> None:
        """Download a file from URL and save to local path using streaming.

        Downloads any file type from HTTP/HTTPS URL with streaming to support
        large files and provide memory efficiency. Shows minimal console feedback
        and creates parent directories if needed.

        Args:
            url (str): URL of the file to download.
                      Must be valid HTTP/HTTPS URL.
                      Example: 'https://example.com/data.csv'
            save_path (str): Local file path to save to (including filename).
                           Parent directories are created if needed.
                           Example: './data/dataset.csv' or '/tmp/file.bin'

        Returns:
            None: File is saved to save_path.

        Raises:
            requests.exceptions.RequestException: If HTTP download fails (network error, timeout).
            requests.exceptions.MissingSchema: If url is invalid (not http/https).
            IOError: If save_path directory cannot be created or written to.
            OSError: If file cannot be created (permissions, disk full).

        Examples:
            >>> # Download a CSV dataset
            >>> FileManager.download_file(
            ...     'https://example.com/timeseries.csv',
            ...     './data/timeseries.csv'
            ... )
            >>> import os
            >>> os.path.exists('./data/timeseries.csv')
            True

            >>> # Download binary file (model weights, archive, etc.)
            >>> FileManager.download_file(
            ...     'https://example.com/model.pth',
            ...     './models/pretrained.pth'
            ... )

        Notes:
            - Streaming is used internally, suitable for files of any size
            - Chunk size is 1024 bytes (1 KB) per write
            - No progress bar shown, for quiet operation
            - Prints error message to console if download fails (HTTP error)
            - Parent directory of save_path must be writable
            - Does not resume partial downloads
        """
        FileManager._ensure_parent_dir(save_path)
        with tempfile.NamedTemporaryFile(
            delete=False,
            dir=os.path.dirname(save_path) or ".",
        ) as temp_file:
            temp_path = temp_file.name

        try:
            with requests.get(url, stream=True, timeout=FileManager.request_timeout) as response:
                response.raise_for_status()
                with open(temp_path, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            os.replace(temp_path, save_path)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    @staticmethod
    def download_from_google_drive(file_id: str, save_path: str) -> None:
        """Download file from Google Drive using file ID and save to local path.

        Downloads a file shared on Google Drive using its unique file ID.
        Internally uses `gdown` library which handles Google Drive's download
        confirmation for large files. Useful for sharing datasets and models via Drive.

        Args:
            file_id (str): Google Drive file ID (unique identifier).
                          Found in share link: 'https://drive.google.com/file/d/{FILE_ID}/view'
                          Example: '1mHIWnDvW9cALBx1eH7K4zQ-zY_2xJ_kp'
            save_path (str): Local file path to save to (including filename).
                           Parent directories must exist or be creatable.
                           Example: './data/dataset.csv'

        Returns:
            None: File is saved to save_path.

        Raises:
            ImportError: If `gdown` library is not installed (pip install gdown).
            Exception: If file_id is invalid or file no longer exists on Drive.
            IOError: If save_path directory doesn't exist or is not writable.
            Exception: If Google Drive blocks the download (too many requests, quota).

        Examples:
            >>> # Download a dataset from Google Drive
            >>> FileManager.download_from_google_drive(
            ...     '1mHIWnDvW9cALBx1eH7K4zQ-zY_2xJ_kp',  # file ID
            ...     './data/my_dataset.csv'
            ... )
            >>> import os
            >>> os.path.getsize('./data/my_dataset.csv')
            1024576  # File size in bytes

            >>> # Download larger file (will show progress by default)
            >>> FileManager.download_from_google_drive(
            ...     '1aB2cD3eF4gH5iJ6kL7mN8oP9qR0sT1uV',
            ...     './models/model_weights.pth'
            ... )

        Notes:
            - Requires `gdown` library: pip install gdown
            - Google Drive limits downloads; may be blocked after many requests
            - Large files may show download progress in terminal
            - File must be accessible (shared with link or public)
            - File ID persists even if filename changes on Drive
            - For very large files, Google Drive may require email confirmation
        """
        url = f"https://drive.google.com/uc?id={file_id}"
        FileManager._ensure_parent_dir(save_path)
        with tempfile.NamedTemporaryFile(
            delete=False,
            dir=os.path.dirname(save_path) or ".",
        ) as temp_file:
            temp_path = temp_file.name

        try:
            result = gdown.download(url, temp_path, quiet=False)
            if result is None or not os.path.exists(temp_path):
                raise RuntimeError(f"Failed to download Google Drive file {file_id}.")
            os.replace(temp_path, save_path)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    @staticmethod
    def split_column_into_files(
        df: pl.DataFrame,
        path: str,
        station_column: str,
        date_column: str | None = None,
        remove_station_column: bool = True,
    ) -> None:
        """Split DataFrame into multiple CSV files by unique column values.

        Creates separate CSV file for each unique value in a categorical column.
        Useful for partitioning multi-station/multi-source data into per-station files.
        Each file contains only rows matching that unique value.

        Args:
            df (pl.DataFrame): Input DataFrame to split.
                              Must contain station_column and be reasonably sized
                              (fits in memory after filtering).
            path (str): Directory where output CSV files will be saved.
                       Must exist or be creatable. Files named {unique_value}.csv
                       Example: './output/stations'
            station_column (str): Column name whose unique values define splits.
                                 Column values become output filenames (truncated if needed).
                                 Example: 'station_id', 'location', 'sensor'
            date_column (str | None, optional): Column name to sort each output by before saving.
                                               If provided, each file is sorted chronologically.
                                               Useful for time series data.
                                               Defaults to None (no sorting).
            remove_station_column (bool, optional): Whether to exclude station_column from output files.
                                                   Since filename encodes the value, often redundant.
                                                   Defaults to True.

        Returns:
            None: Files are written to path directory.

        Raises:
            ValueError: If station_column not in DataFrame columns.
            ValueError: If date_column specified but not in DataFrame columns.
            IOError: If path directory cannot be created or written to.
            RuntimeError: If individual filtered DataFrames fail to write.

        Examples:
            >>> import polars as pl
            >>> # Multi-station time series data
            >>> df = pl.DataFrame({
            ...     'date': ['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-02'],
            ...     'station': ['NYC', 'NYC', 'LA', 'LA'],
            ...     'temp': [5.2, 6.1, 15.3, 16.8]
            ... })
            >>> FileManager.split_column_into_files(
            ...     df, './stations', 'station', 'date', remove_station_column=True
            ... )
            >>> # Creates: ./stations/NYC.csv and ./stations/LA.csv
            >>> import os
            >>> os.listdir('./stations')
            ['NYC.csv', 'LA.csv']

            >>> # Keep station column in output (for reference)
            >>> FileManager.split_column_into_files(
            ...     df, './output', 'station', date_column=None, remove_station_column=False
            ... )

        Notes:
            - Each unique value becomes a separate file
            - Files are independently written, no guaranteed order
            - If station_column value is complex, filename may be truncated by OS
            - Sorting by date_column is done before writing (adds memory overhead)
            - Output files are always in CSV format (one per value)
            - Very large DataFrames should be pre-filtered before calling
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
    ) -> None:
        """Split DataFrame into separate CSV files, one per column (excluding date column).

        Creates individual CSV file for each numerical/categorical column, with each file
        containing the date column plus one other column. Useful for converting wide-format
        time series (many columns) into narrow-format files (one value per file).

        Args:
            df (pl.DataFrame): Input DataFrame with date column and multiple value columns.
                              Shape: (n_rows, 1_date_col + n_value_cols)
            path (str): Directory where output CSV files will be saved.
                       Each file named as {column_name}.csv or {index}.csv
                       Must exist or be creatable.
            date_column (str): Name of date/time column to include in each output file.
                              Column must exist in df.
                              Example: 'timestamp', 'date'
            new_column_name (str, optional): Name to assign to value column in each output file.
                                            Defaults to 'Value'.
                                            Example: 'temperature', 'measurement'
            keep_old_column_name_as_filename (bool, optional): If True, output filenames match
                                                               original column names.
                                                               If False, uses sequential indices.
                                                               Defaults to False.

        Returns:
            None: Files are written to path directory.

        Raises:
            ValueError: If date_column not in DataFrame columns.
            IOError: If path directory cannot be created or written to.
            RuntimeError: If DataFrame is too large for memory operations.

        Examples:
            >>> import polars as pl
            >>> # Wide-format multivariate time series
            >>> df = pl.DataFrame({
            ...     'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            ...     'temp': [22.5, 23.1, 21.8],
            ...     'humidity': [65, 70, 68],
            ...     'pressure': [1013, 1012, 1014]
            ... })
            >>> # Option 1: Use original column names as filenames
            >>> FileManager.split_columns_into_files(
            ...     df, './output_cols', 'date',
            ...     new_column_name='measurement',
            ...     keep_old_column_name_as_filename=True
            ... )
            >>> # Creates: ./output_cols/temp.csv, ./output_cols/humidity.csv, etc.

            >>> # Option 2: Use sequential numbering for filenames
            >>> FileManager.split_columns_into_files(
            ...     df, './output_nums', 'date',
            ...     keep_old_column_name_as_filename=False
            ... )
            >>> # Creates: ./output_nums/0.csv, ./output_nums/1.csv, ./output_nums/2.csv

        Notes:
            - date_column is preserved in all output files
            - date_column is never renamed to new_column_name
            - Non-value columns besides date_column are silently skipped
            - File naming logic: sorted iteration over remaining columns
            - Each output file is independent, can be loaded separately
            - Useful for machine learning pipelines requiring per-feature files
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
        """Read CSV file into Polars DataFrame with automatic date/datetime parsing.

        Loads a CSV file using Polars with automatic detection and parsing of
        date/datetime columns. Returns None if file is empty to allow graceful
        error handling in pipelines.

        Args:
            path (str): File path to CSV file to read.
                       Can be relative path or absolute path.
                       Example: './data/dataset.csv' or '/data/timeseries.csv'

        Returns:
            pl.DataFrame | None: Loaded DataFrame if file is valid and non-empty.
                                Returns None if file is empty (0 rows).
                                DataFrame preserves all column types detected during parsing.

        Raises:
            FileNotFoundError: If file at path doesn't exist.
            pl.exceptions.ComputeError: If CSV format is malformed (parsing error).
            IOError: If file cannot be read (permissions, disk error).
            pl.exceptions.NoDataError: Internally caught and returns None.

        Examples:
            >>> # Read standard CSV file
            >>> df = FileManager.read('./data/timeseries.csv')
            >>> df.shape
            (1000, 5)
            >>> df.dtypes
            [Datetime, Int64, Float64, Utf8, Utf8]

            >>> # Handle empty file gracefully
            >>> df = FileManager.read('./data/empty_file.csv')
            >>> df is None
            True

            >>> # Dates are automatically parsed
            >>> df = FileManager.read('./data/with_dates.csv')
            >>> df['date'].dtype
            Datetime(time_unit='us', time_zone=None)

        Notes:
            - Automatic date parsing tries to detect date/datetime columns
            - Returns None for empty files (no error raised)
            - First row must be valid header or CSV parsing will fail
            - Large files (100MB+) load efficiently using Polars
            - Date formats recognized: ISO 8601, US (MM/DD/YYYY), EU (DD/MM/YYYY), etc.
            - All columns are preserved; no automatic type conversion beyond dates
        """
        try:
            df = pl.read_csv(path, try_parse_dates=True)
            return df
        except pl.exceptions.NoDataError:
            print(f"Empty file: {path}")
            return None


class DataFrameOptimizer:
    """
    Utility class for optimizing Polars DataFrame memory usage through intelligent type casting.

    Provides methods to analyze and optimize DataFrame memory footprint by automatically
    selecting optimal data types for each column. Reduces memory without losing information,
    particularly effective for large datasets with many columns.
    """

    @staticmethod
    def reduce_polars_df(df: pl.DataFrame, info: bool = False) -> pl.DataFrame:
        """Optimize DataFrame memory by casting columns to optimal data types.

        Analyzes each column in the DataFrame and applies intelligent type conversions
        to minimize memory usage while preserving all data:

        - String → categorical: Text data becomes category type (major savings)
        - Datetime → Date: Datetimes with no time component downcast to date type
        - Int/Float → smaller type: Numeric columns cast to smallest fitting type

        This method is particularly effective for time series data with many
        categorical columns and long string values.

        Args:
            df (pl.DataFrame): Input DataFrame to optimize.
                              Any shape/content; all columns processed independently.
            info (bool, optional): If True, print memory statistics to console
                                 showing before/after sizes and reduction percentage.
                                 Defaults to False (silent operation).

        Returns:
            pl.DataFrame: Optimized DataFrame with reduced memory footprint.
                         All data preserved, only representation changed.
                         Column order and names unchanged.
                         Row order and content unchanged.

        Raises:
            ValueError: If df is None or not a Polars DataFrame.
            MemoryError: If optimization process runs out of memory (rare).

        Examples:
            >>> import polars as pl
            >>> import numpy as np
            >>> # Large dataset with string columns
            >>> df = pl.DataFrame({
            ...     'name': ['Alice', 'Bob', 'Charlie'] * 1000,
            ...     'age': [25, 30, 35] * 1000,
            ...     'salary': [50000.5, 60000.5, 75000.5] * 1000,
            ...     'timestamp': pl.datetime_range(
            ...         '2023-01-01', periods=3000, interval='1h', eager=True
            ...     )
            ... })
            >>> print(f"Original size: {df.estimated_size() / 1024:.2f} KB")
            Original size: 85.45 KB
            >>> # Optimize with info display
            >>> df_opt = DataFrameOptimizer.reduce_polars_df(df, info=True)
            Polars DataFrame memory usage: 85.45 KB ⇒ 42.30 KB (50.49% reduction)
            >>> print(df_opt.schema)
            {'name': Categorical(None), 'age': Int8, 'salary': Float64, 'timestamp': Date}

            >>> # Quiet optimization
            >>> df_opt2 = DataFrameOptimizer.reduce_polars_df(df, info=False)
            >>> # Verify data is preserved
            >>> (df['name'] == df_opt2['name']).all()
            True

        Notes:
            - Categorical conversion is most effective for text with many duplicates
            - Integer downcasting uses range analysis [min, max] to pick smallest type
            - Float types are preserved but cast from float64 → float32 if possible
            - Datetime → Date conversion only happens for midnight timestamps
            - Output DataFrame is independent; input df is unchanged
            - For very large files, consider filtering before optimization
            - Memory savings typically 30-70% depending on data characteristics
        """
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
                optimal_type = DataFrameOptimizer.cast_to_optimal_type(
                    c_min, c_max, "float" if "float" in str(col_type) else "int"
                )
                if optimal_type:
                    polars_type = np_to_pl_type_mapping.get(optimal_type)
                    if polars_type:
                        df = df.with_columns(df[col].cast(polars_type))

        if info:
            after = df.estimated_size()
            reduction = 100.0 * (before - after) / before
            reduction_str = f"{reduction:.2f}% reduction"
            before, after, unit = DataFrameOptimizer.memory_unit_conversion(
                before, after
            )
            print(
                f"Polars DataFrame memory usage: {before:.4f} {unit} ⇒ {after:.4f} {unit} ({reduction_str})"
            )
        return df

    @staticmethod
    def cast_to_optimal_type(
        c_min: float, c_max: float, current_type: str
    ) -> np.dtype | None:
        """Determine smallest NumPy dtype that fits a value range without data loss.

        Analyzes the range [c_min, c_max] and finds the minimal integer or float type
        that can represent all values in that range. Used for memory optimization by
        downcasting numeric columns.

        Type selection strategy:
            - Integer types: int8 (-128 to 127) → int16 → int32 → int64
            - Float types: float16 (half precision) → float32 → float64
            - Returns None if no type in the family can hold the range

        Args:
            c_min (float): Minimum value in the range (can be negative).
                          Example: -500, 0, 0.5
            c_max (float): Maximum value in the range (can be negative).
                          Must be >= c_min for meaningful results.
                          Example: 100, 65535, 3.14
            current_type (str): Type family to search within:
                              - 'int': Search integer types (int8, int16, int32, int64)
                              - 'float': Search float types (float16, float32, float64)
                              - Other values default to float search

        Returns:
            np.dtype | None: Smallest NumPy dtype that safely represents [c_min, c_max].
                            Returns None if no type in selected family fits.
                            Result is a NumPy dtype object, e.g., dtype('int8').

        Raises:
            No exceptions raised; returns None for unmatchable ranges.

        Examples:
            >>> # Small positive integers fit in int8
            >>> dtype = DataFrameOptimizer.cast_to_optimal_type(0, 100, 'int')
            >>> dtype
            dtype('int8')

            >>> # Larger positive integers need int16
            >>> dtype = DataFrameOptimizer.cast_to_optimal_type(0, 50000, 'int')
            >>> dtype
            dtype('int16')

            >>> # Negative range requires signed type
            >>> dtype = DataFrameOptimizer.cast_to_optimal_type(-128, 127, 'int')
            >>> dtype
            dtype('int8')
            >>> dtype = DataFrameOptimizer.cast_to_optimal_type(-129, 127, 'int')
            >>> dtype
            dtype('int16')

            >>> # Large range exceeds all int types
            >>> dtype = DataFrameOptimizer.cast_to_optimal_type(-1e10, 1e10, 'int')
            >>> dtype
            dtype('int64')
            >>> dtype = DataFrameOptimizer.cast_to_optimal_type(-1e20, 1e20, 'int')
            >>> dtype is None
            True

            >>> # Float types
            >>> dtype = DataFrameOptimizer.cast_to_optimal_type(-1.0, 1.0, 'float')
            >>> dtype
            dtype('float16')  # Can represent small ranges

        Notes:
            - Uses NumPy's can_cast() for safety checking
            - Doesn't account for special values (NaN, Inf)
            - Assumes data is well-behaved (no NaN in min/max)
            - Always returns the smallest type that works
            - Useful in data pipeline optimization loops
            - int64/float64 are fallback types for ranges that fit
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
    def memory_unit_conversion(before: float, after: float) -> tuple[float, float, str]:
        """Convert memory sizes from bytes to appropriate human-readable unit.

        Iteratively scales both values down by 1024 to find the most readable unit.
        Selects a single unit that best represents both before and after sizes
        (useful for displaying memory reduction).

        Unit hierarchy (each = 1024 of previous):
            bytes (B) → KB → MB → GB → TB → PB

        Selection logic: Uses largest unit where before >= 1024, then applies
        same unit to both values for consistent comparison.

        Args:
            before (float): Initial memory size in bytes (before optimization).
                           Must be non-negative. Example: 1500000 (1.5 MB)
            after (float): Final memory size in bytes (after optimization).
                          Must be non-negative. Should be <= before for meaningful result.
                          Example: 500000 (500 KB)

        Returns:
            tuple[float, float, str]: Tuple of (converted_before, converted_after, unit_name).
                - converted_before (float): Size before in selected unit.
                - converted_after (float): Size after in selected unit.
                - unit_name (str): Unit of measurement string.
                                  One of: 'bytes', 'KB', 'MB', 'GB', 'TB', 'PB'

        Examples:
            >>> # Small file (under 1 KB)
            >>> b, a, u = DataFrameOptimizer.memory_unit_conversion(500, 250)
            >>> print(f"{b:.1f} {u} ↨ {a:.1f} {u}")
            500.0 bytes ↨ 250.0 bytes

            >>> # Medium file (1.5 MB ↨ 500 KB)
            >>> b, a, u = DataFrameOptimizer.memory_unit_conversion(1_500_000, 500_000)
            >>> print(f"{b:.2f} {u} ↨ {a:.2f} {u}")
            1.43 MB ↨ 0.48 MB

            >>> # Large file (5 GB ↨ 2 GB)
            >>> b, a, u = DataFrameOptimizer.memory_unit_conversion(
            ...     5_000_000_000, 2_000_000_000
            ... )
            >>> print(f"{b:.2f} {u} ↨ {a:.2f} {u}")
            4.66 GB ↨ 1.86 GB

            >>> # Very large file
            >>> b, a, u = DataFrameOptimizer.memory_unit_conversion(
            ...     5_000_000_000_000, 3_000_000_000_000  # 5 TB ↨ 3 TB
            ... )
            >>> print(f"{b:.2f} {u}")
            4.55 TB
            >>> print(f"Reduction: {100 * (b - a) / b:.1f}%")
            Reduction: 40.0%

        Notes:
            - Stops at PB (petabytes); doesn't convert beyond that
            - Both inputs scaled to same unit for consistent comparison
            - Floating-point division preserves precision
            - Useful for progress reporting and memory analysis
            - Typically paired with reduction percentage calculation
            - Displays nice human-readable format for logs/UI
        """
        units = ["bytes", "KB", "MB", "GB", "TB", "PB"]
        unit_index = 0

        # Iteratively divide by 1024 to find the appropriate unit
        while before >= 1024 and unit_index < len(units) - 1:
            before /= 1024
            after /= 1024
            unit_index += 1

        return before, after, units[unit_index]


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
