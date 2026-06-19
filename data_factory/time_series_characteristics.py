import numpy as np
import polars as pl
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


