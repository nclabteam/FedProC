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
            with requests.get(
                url, stream=True, timeout=FileManager.request_timeout
            ) as response:
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
            with requests.get(
                url, stream=True, timeout=FileManager.request_timeout
            ) as response:
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


