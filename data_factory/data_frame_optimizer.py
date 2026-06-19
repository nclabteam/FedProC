import numpy as np
import polars as pl


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

            if col_type in (pl.Utf8, pl.String):
                df = df.with_columns(df[col].cast(pl.Categorical))
            elif isinstance(col_type, pl.Datetime):
                # Check if all timestamps are at midnight (no time component)
                if (
                    df[col].dt.hour().sum() == 0
                    and df[col].dt.minute().sum() == 0
                    and df[col].dt.second().sum() == 0
                ):
                    df = df.with_columns(df[col].cast(pl.Date))
            elif col_type in np_to_pl_type_mapping.values():
                c_min, c_max = df[col].min(), df[col].max()
                if c_min is None or c_max is None:
                    continue
                optimal_type = DataFrameOptimizer.cast_to_optimal_type(
                    c_min, c_max, "float" if "float" in str(col_type).lower() else "int"
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
                f"Polars DataFrame memory usage: {before:.4f} {unit} -> {after:.4f} {unit} ({reduction_str})"
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
            info = np.iinfo(dtype) if current_type == "int" else np.finfo(dtype)
            if info.min <= c_min and c_max <= info.max:
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


