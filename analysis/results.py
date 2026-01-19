"""
Analysis module for aggregating federated learning experiment results.

This module provides functionality to:
- Load experiment configurations and results from a runs directory
- Compute bandwidth statistics (downlink, uplink, total) per experiment
- Aggregate metrics across multiple runs using various aggregation modes
- Filter experiments by model, strategy, dataset, output length, or name
- Convert time and size units for display
- Display results in formatted polars tables
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import polars as pl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from strategies.base import SharedMethods

# =============================================================================
# Logging Configuration
# =============================================================================
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# Polars Display Configuration
# =============================================================================
pl.Config.set_tbl_cols(100)  # Maximum number of columns to display
pl.Config.set_tbl_rows(100)  # Maximum number of rows to display

# =============================================================================
# Type Definitions and Constants
# =============================================================================

# Aggregation modes for per-run statistics
AGG_MODES: Tuple[str, ...] = ("min", "max", "mean", "last", "median")
AggMode = Literal["min", "max", "mean", "last", "median"]

# Time unit options (base unit: seconds)
TIME_UNITS: Tuple[str, ...] = ("s", "ms", "m", "h")
TimeUnit = Literal["s", "ms", "m", "h"]

# Size unit options (base unit: MB)
SIZE_UNITS: Tuple[str, ...] = ("b", "kb", "mb", "gb", "tb")
SizeUnit = Literal["b", "kb", "mb", "gb", "tb"]

# Pivot options for table display
PIVOT_OPTIONS: Tuple[str, ...] = ("model", "strategy")
PivotOption = Literal["model", "strategy"]

# Output value when no valid data exists
MISSING_VALUE: float = 0.0

# Time columns that should be converted
TIME_COLUMNS: Tuple[str, ...] = ("efficiency",)

# Size columns that should be converted
SIZE_COLUMNS: Tuple[str, ...] = ("downlink", "uplink", "communication")

# Available metrics for display
METRICS: Tuple[str, ...] = (
    "all",
    "loss",
    "efficiency",
    "communication",
    "last_improvement_round",
    "longest_improvement_streak",
    "most_frequent_improvement_streak",
    "oscillation_count",
    "improvement_ratio",
    "improvement_magnitude",
)

# Metric descriptions for display
METRIC_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "loss": {
        "title": "Test loss (model performance)",
        "explanation": "Average test loss across all clients or global model performance.\n"
        "Lower values = Better model performance\n"
        "Higher values = Worse model performance\n"
        "Primary metric for evaluating model quality and convergence.",
    },
    "efficiency": {
        "title": "Time per training round (computational efficiency)",
        "explanation": "Average time taken to complete one federated learning round.\n"
        "Lower values = Faster training, more efficient system\n"
        "Higher values = Slower training, resource bottlenecks\n"
        "Critical for assessing practical deployment feasibility.",
    },
    "communication": {
        "title": "Total communication cost (bandwidth usage)",
        "explanation": "Combined uplink and downlink data transfer per round.\n"
        "Lower values = Less bandwidth usage, lower cost\n"
        "Higher values = Higher bandwidth requirements and costs\n"
        "Key metric for federated learning scalability and deployment cost.",
    },
    "last_improvement_round": {
        "title": "Round with the last improvement (convergence indicator)",
        "explanation": "The last training round where the model achieved better loss than before.\n"
        "Lower values = Model peaked early, may need more training\n"
        "Higher values = Model continued improving throughout training\n"
        "Indicates when the model stopped making progress (convergence point).",
    },
    "longest_improvement_streak": {
        "title": "Maximum consecutive rounds of improvement (stability measure)",
        "explanation": "The longest sequence of consecutive rounds with meaningful loss reduction.\n"
        "Lower values = Erratic training with frequent plateaus or oscillations\n"
        "Higher values = Stable, consistent optimization progress\n"
        "Reflects training smoothness and optimization algorithm effectiveness.",
    },
    "most_frequent_improvement_streak": {
        "title": "Most common improvement streak length (training pattern)",
        "explanation": "The streak length that occurred most often during training.\n"
        "Lower values = Training progresses in short bursts with frequent stagnation\n"
        "Higher values = Sustained improvement patterns dominate the training\n"
        "Reveals the typical learning rhythm and optimization behavior.",
    },
    "oscillation_count": {
        "title": "Number of loss direction changes (instability measure)",
        "explanation": "Counts how many times loss direction changed (up/down transitions).\n"
        "Lower values = Stable, monotonic improvement (ideal)\n"
        "Higher values = Unstable training, possible learning rate or optimization issues\n"
        "High oscillations suggest need for learning rate adjustment or different optimizer.",
    },
    "improvement_ratio": {
        "title": "Fraction of rounds with improvement (training efficiency: 0.0-1.0)",
        "explanation": "Proportion of training rounds that resulted in loss reduction.\n"
        "Lower values = Inefficient training, many wasted rounds\n"
        "Higher values = Efficient training, most rounds contributed to learning\n"
        "Values >0.5 indicate productive training; <0.3 suggests optimization problems.",
    },
    "improvement_magnitude": {
        "title": "Average loss reduction per improvement (improvement speed)",
        "explanation": "Average magnitude of loss reduction when improvements occur.\n"
        "Calculated as average of all loss deltas where loss[i] < loss[i-1].\n"
        "Higher values = Larger improvements in each step (better optimizer efficiency)\n"
        "Lower values = Marginal improvements (slower convergence progress).",
    },
}


class Analysis:
    """
    Analyze federated learning experiment results.

    This class reads experiment data from a runs directory structure:
        runs_dir/
            experiment_name/
                run_0/
                    results/
                        server.csv
                        client_000.csv
                        client_001.csv
                        ...
                run_1/
                    ...
                config.json
                results.csv

    Attributes:
        runs_dir: Path to the runs directory.
        max_lines: Maximum number of lines to read from CSV files (None for all).
        std_multiplier: Factor to multiply standard deviation values.
        decimal_places: Number of decimal places to round output values.
        agg_mode: Aggregation mode for per-run statistics.
        time_unit: Output unit for time values.
        size_unit: Output unit for size values.
        input_sentinel: Value indicating invalid/missing input data.
        output_sentinel: Value to use in output when no valid data exists.
    """

    # =========================================================================
    # Unit conversion (static methods)
    # =========================================================================

    @staticmethod
    def convert_time(value: float, from_unit: str = "s", to_unit: str = "s") -> float:
        """
        Convert time value between units.

        Args:
            value: Time value to convert.
            from_unit: Source unit ('s', 'ms', 'm', 'h').
            to_unit: Target unit ('s', 'ms', 'm', 'h').

        Returns:
            Converted time value.
        """
        to_seconds: Dict[str, float] = {
            "ms": 0.001,
            "s": 1.0,
            "m": 60.0,
            "h": 3600.0,
        }
        from_seconds: Dict[str, float] = {
            "ms": 1000.0,
            "s": 1.0,
            "m": 1.0 / 60.0,
            "h": 1.0 / 3600.0,
        }

        from_unit = from_unit.lower()
        to_unit = to_unit.lower()

        if from_unit not in to_seconds:
            raise ValueError(f"Unknown time unit: {from_unit}")
        if to_unit not in from_seconds:
            raise ValueError(f"Unknown time unit: {to_unit}")

        seconds = value * to_seconds[from_unit]
        return seconds * from_seconds[to_unit]

    @staticmethod
    def convert_size(value: float, from_unit: str = "mb", to_unit: str = "mb") -> float:
        """
        Convert size value between units.

        Args:
            value: Size value to convert.
            from_unit: Source unit ('b', 'kb', 'mb', 'gb', 'tb').
            to_unit: Target unit ('b', 'kb', 'mb', 'gb', 'tb').

        Returns:
            Converted size value.
        """
        to_bytes: Dict[str, float] = {
            "b": 1.0,
            "kb": 1024.0,
            "mb": 1024.0**2,
            "gb": 1024.0**3,
            "tb": 1024.0**4,
        }
        from_bytes: Dict[str, float] = {
            "b": 1.0,
            "kb": 1.0 / 1024.0,
            "mb": 1.0 / (1024.0**2),
            "gb": 1.0 / (1024.0**3),
            "tb": 1.0 / (1024.0**4),
        }

        from_unit = from_unit.lower()
        to_unit = to_unit.lower()

        if from_unit not in to_bytes:
            raise ValueError(f"Unknown size unit: {from_unit}")
        if to_unit not in from_bytes:
            raise ValueError(f"Unknown size unit: {to_unit}")

        bytes_val = value * to_bytes[from_unit]
        return bytes_val * from_bytes[to_unit]

    @staticmethod
    def resolve_loss_metric(save_local_model: bool) -> str:
        """
        Resolve the loss metric name based on save_local_model flag.

        Args:
            save_local_model: Whether local models are saved.

        Returns:
            'personal_avg_test_loss' if save_local_model else 'global_avg_test_loss'.
        """
        return "personal_avg_test_loss" if save_local_model else "global_avg_test_loss"

    def __init__(
        self,
        runs_dir: str | Path = "runs",
        output_dir: str | Path = "analysis/tables",
        max_lines: Optional[int] = None,
        std_multiplier: float = 1.0,
        decimal_places: int = 4,
        agg_mode: AggMode = "min",
        time_unit: TimeUnit = "s",
        size_unit: SizeUnit = "mb",
    ) -> None:
        """
        Initialize the Analysis class.

        Args:
            runs_dir: Path to the runs directory.
            output_dir: Path to the output directory for tables.
            max_lines: Maximum number of lines to read from CSV files.
            std_multiplier: Factor to multiply standard deviation values.
            decimal_places: Number of decimal places to round output values.
            agg_mode: Aggregation mode ('min', 'max', 'mean', 'last', 'median').
            time_unit: Output unit for time values ('s', 'ms', 'm', 'h').
            size_unit: Output unit for size values ('b', 'kb', 'mb', 'gb', 'tb').

        Raises:
            ValueError: If agg_mode, time_unit, or size_unit is not valid.
        """
        if agg_mode not in AGG_MODES:
            raise ValueError(f"agg_mode must be one of {AGG_MODES}, got '{agg_mode}'")
        if time_unit.lower() not in TIME_UNITS:
            raise ValueError(
                f"time_unit must be one of {TIME_UNITS}, got '{time_unit}'"
            )
        if size_unit.lower() not in SIZE_UNITS:
            raise ValueError(
                f"size_unit must be one of {SIZE_UNITS}, got '{size_unit}'"
            )

        self.runs_dir: Path = Path(runs_dir)
        self.output_dir: Path = Path(output_dir)
        self.max_lines: Optional[int] = max_lines
        self.std_multiplier: float = std_multiplier
        self.decimal_places: int = decimal_places
        self.agg_mode: AggMode = agg_mode
        self.time_unit: TimeUnit = time_unit.lower()  # type: ignore
        self.size_unit: SizeUnit = size_unit.lower()  # type: ignore
        self.input_sentinel: float = SharedMethods.default_value
        self.output_sentinel: float = MISSING_VALUE

        logger.debug(
            "Analysis initialized: runs_dir=%s, agg_mode=%s, time_unit=%s, size_unit=%s",
            self.runs_dir,
            self.agg_mode,
            self.time_unit,
            self.size_unit,
        )

    # =========================================================================
    # Path helpers
    # =========================================================================

    def _config_path(self, experiment_name: str) -> Path:
        """Get the path to an experiment's config.json file."""
        return self.runs_dir / experiment_name / "config.json"

    def _results_path(self, experiment_name: str) -> Path:
        """Get the path to an experiment's results.csv file."""
        return self.runs_dir / experiment_name / "results.csv"

    # =========================================================================
    # File I/O
    # =========================================================================

    def _read_json(self, path: Path) -> Optional[Dict]:
        """
        Read a JSON file and return its contents as a dictionary.

        Args:
            path: Path to the JSON file.

        Returns:
            Dictionary containing the JSON data, or None if reading fails.
        """
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logger.debug("Failed to load JSON: %s", path)
            return None

    def _read_csv(self, path: Path) -> Optional[Dict[str, List]]:
        """
        Read a CSV file and return its contents as a dictionary of lists.

        Args:
            path: Path to the CSV file.

        Returns:
            Dictionary mapping column names to lists of values, or None if reading fails.
        """
        if not path.exists():
            return None
        try:
            df = (
                pl.read_csv(path, n_rows=self.max_lines)
                if self.max_lines
                else pl.read_csv(path)
            )
            return df.to_dict(as_series=False)
        except Exception as e:
            logger.warning("Failed to read CSV %s: %s", path, e)
            return None

    def _read_server_csv(self, run_dir: Path) -> Optional[Dict[str, List]]:
        """Read the server.csv file from a run directory."""
        return self._read_csv(run_dir / "results" / "server.csv")

    def _read_client_csvs(self, run_dir: Path) -> List[Dict[str, List]]:
        """
        Read all client CSV files from a run directory.

        Client files are expected to be named client_000.csv, client_001.csv, etc.

        Args:
            run_dir: Path to the run directory.

        Returns:
            List of dictionaries, one per client CSV file.
        """
        results_dir = run_dir / "results"
        if not results_dir.exists():
            return []

        clients: List[Dict[str, List]] = []
        for f in sorted(results_dir.glob("client_*.csv")):
            data = self._read_csv(f)
            if data is not None:
                clients.append(data)
        return clients

    # =========================================================================
    # Data parsing and validation
    # =========================================================================

    def _is_valid_input(self, value: float) -> bool:
        """Check if a value is valid (not the input sentinel)."""
        return value != self.input_sentinel

    def _parse_numeric_list(self, seq: List, exclude_zero: bool = False) -> List[float]:
        """
        Parse a list to extract valid numeric values.

        Args:
            seq: List of values to parse.
            exclude_zero: If True, exclude zero values from the result.

        Returns:
            List of valid numeric values.
        """
        vals: List[float] = []
        for x in seq:
            try:
                xv = float(x)
            except (ValueError, TypeError):
                continue
            if xv == self.input_sentinel:
                continue
            if exclude_zero and xv == 0:
                continue
            vals.append(xv)
        return vals

    # =========================================================================
    # Instance unit conversion helpers
    # =========================================================================

    def _convert_time_value(self, value: float) -> float:
        """Convert time value from seconds to configured time unit."""
        return self.convert_time(value, from_unit="s", to_unit=self.time_unit)

    def _convert_size_value(self, value: float) -> float:
        """Convert size value from MB to configured size unit."""
        return self.convert_size(value, from_unit="mb", to_unit=self.size_unit)

    def _is_time_column(self, col_name: str) -> bool:
        """Check if a column should be treated as a time value."""
        return col_name in TIME_COLUMNS

    def _is_size_column(self, col_name: str) -> bool:
        """Check if a column should be treated as a size value."""
        return col_name in SIZE_COLUMNS

    # =========================================================================
    # Formatting
    # =========================================================================

    def _format_mean(self, value: float) -> float:
        """Round a mean value to the configured decimal places."""
        return round(value, self.decimal_places)

    def _format_std(self, value: float) -> float:
        """Scale and round a standard deviation value."""
        return round(value * self.std_multiplier, self.decimal_places)

    def _format_mean_std_str(self, mean: float, std: float) -> str:
        """
        Format mean and std as a display string.

        Args:
            mean: Mean value (already rounded via _format_mean).
            std: Std value (already scaled and rounded via _format_std).

        Returns:
            String in format "mean±std".
        """
        return f"{mean}±{std}"

    # =========================================================================
    # Aggregation
    # =========================================================================

    def _compute_per_run_agg(self, vals: List[float]) -> Optional[float]:
        """
        Compute a single aggregate value from a list of values.

        Args:
            vals: List of numeric values.

        Returns:
            The aggregated value based on agg_mode, or None if vals is empty.
        """
        if not vals:
            return None

        if self.agg_mode == "min":
            return min(vals)
        elif self.agg_mode == "max":
            return max(vals)
        elif self.agg_mode == "mean":
            return float(np.mean(vals))
        elif self.agg_mode == "last":
            return vals[-1]
        elif self.agg_mode == "median":
            return float(np.median(vals))
        return None

    def _compute_aggregates(
        self,
        per_run_values: List[float],
        key_prefix: str,
        is_time: bool = False,
        is_size: bool = False,
    ) -> Dict[str, float]:
        """
        Compute mean and standard deviation across runs.

        Args:
            per_run_values: List of per-run aggregate values.
            key_prefix: Prefix for the output keys.
            is_time: If True, convert values from seconds to time_unit.
            is_size: If True, convert values from MB to size_unit.

        Returns:
            Dictionary with keys '{prefix}_{agg_mode}_mean' and '{prefix}_{agg_mode}_std'.
        """
        result: Dict[str, float] = {}
        mean_key = f"{key_prefix}_{self.agg_mode}_mean"
        std_key = f"{key_prefix}_{self.agg_mode}_std"

        if per_run_values:
            mean_val = float(np.mean(per_run_values))
            std_val = float(np.std(per_run_values, ddof=0))

            if is_time:
                mean_val = self._convert_time_value(mean_val)
                std_val = self._convert_time_value(std_val)
            elif is_size:
                mean_val = self._convert_size_value(mean_val)
                std_val = self._convert_size_value(std_val)

            result[mean_key] = self._format_mean(mean_val)
            result[std_key] = self._format_std(std_val)
        else:
            result[mean_key] = self.output_sentinel
            result[std_key] = self.output_sentinel

        return result

    # =========================================================================
    # Bandwidth statistics
    # =========================================================================

    def _compute_bandwidth_stats(self, run_dir: Path) -> Dict[str, List[float]]:
        """
        Compute per-round bandwidth statistics for a single run.

        Bandwidth is computed as:
        - downlink: Server's send_mb (server -> clients)
        - uplink: Sum of all clients' send_mb (clients -> server)
        - total: downlink + uplink

        Args:
            run_dir: Path to the run directory.

        Returns:
            Dictionary with 'downlink', 'uplink', and 'total' lists.
            Each list contains valid per-round values (sentinel values excluded).
            Values are in MB (conversion happens during aggregation).
        """
        server_data = self._read_server_csv(run_dir)
        client_datas = self._read_client_csvs(run_dir)

        downlink_vals: List[float] = []
        if server_data and "send_mb" in server_data:
            downlink_vals = self._parse_numeric_list(
                server_data["send_mb"], exclude_zero=True
            )

        num_rounds = 0
        if server_data and "send_mb" in server_data:
            num_rounds = len(server_data["send_mb"])
        for cd in client_datas:
            if "send_mb" in cd:
                num_rounds = max(num_rounds, len(cd["send_mb"]))

        uplink_per_round: List[float] = []
        for i in range(num_rounds):
            round_sum = 0.0
            has_valid = False

            for cd in client_datas:
                if "send_mb" not in cd or i >= len(cd["send_mb"]):
                    continue
                try:
                    xv = float(cd["send_mb"][i])
                    if xv != self.input_sentinel:
                        round_sum += xv
                        has_valid = True
                except (ValueError, TypeError):
                    continue

            if has_valid and round_sum > 0:
                uplink_per_round.append(round_sum)

        total_per_round: List[float] = []
        if server_data and "send_mb" in server_data:
            for i in range(num_rounds):
                downlink_i: Optional[float] = None
                if i < len(server_data["send_mb"]):
                    try:
                        dv = float(server_data["send_mb"][i])
                        if dv != self.input_sentinel and dv > 0:
                            downlink_i = dv
                    except (ValueError, TypeError):
                        pass

                uplink_i = 0.0
                uplink_valid = False
                for cd in client_datas:
                    if "send_mb" not in cd or i >= len(cd["send_mb"]):
                        continue
                    try:
                        uv = float(cd["send_mb"][i])
                        if uv != self.input_sentinel:
                            uplink_i += uv
                            uplink_valid = True
                    except (ValueError, TypeError):
                        continue

                if downlink_i is not None and uplink_valid and uplink_i > 0:
                    total_per_round.append(downlink_i + uplink_i)

        return {
            "downlink": downlink_vals,
            "uplink": uplink_per_round,
            "total": total_per_round,
        }

    # =========================================================================
    # Loss-derived metric helpers
    # =========================================================================

    def _compute_last_improvement_round(self, vals: List[float]) -> Optional[float]:
        """Compute the last round where loss improved."""
        if len(vals) < 2:
            return None

        best_so_far = vals[0]
        last_imp_idx = 0
        for i in range(1, len(vals)):
            if vals[i] < best_so_far:
                last_imp_idx = i + 1  # 1-based
                best_so_far = vals[i]

        return float(last_imp_idx) if last_imp_idx > 0 else None

    def _compute_improvement_streaks(
        self, vals: List[float]
    ) -> Tuple[Optional[float], Optional[float]]:
        """Compute longest and most frequent improvement streaks.

        Returns:
            Tuple of (longest_streak, most_frequent_streak)
        """
        if len(vals) < 2:
            return None, None

        best_so_far = vals[0]
        improvements: List[bool] = []
        for i in range(1, len(vals)):
            if vals[i] < best_so_far:
                improvements.append(True)
                best_so_far = vals[i]
            else:
                improvements.append(False)

        # Compute streaks
        max_streak = 0
        cur_streak = 0
        streaks: List[int] = []
        for imp in improvements:
            if imp:
                cur_streak += 1
            else:
                if cur_streak > 0:
                    streaks.append(cur_streak)
                max_streak = max(max_streak, cur_streak)
                cur_streak = 0
        if cur_streak > 0:
            streaks.append(cur_streak)
            max_streak = max(max_streak, cur_streak)

        longest = float(max_streak) if max_streak > 0 else None

        # Most frequent
        most_frequent = None
        if streaks:
            cnt = Counter(streaks)
            most_common = cnt.most_common()
            max_freq = most_common[0][1]
            candidates = [length for length, freq in most_common if freq == max_freq]
            most_frequent = float(max(candidates))

        return longest, most_frequent

    def _compute_oscillation_count(self, vals: List[float]) -> Optional[float]:
        """Compute number of direction changes in loss."""
        if len(vals) < 2:
            return None

        deltas = [vals[i] - vals[i - 1] for i in range(1, len(vals))]
        signs = [0 if d == 0 else (1 if d > 0 else -1) for d in deltas if d != 0]

        if len(signs) < 2:
            return 0.0

        oscillations = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
        return float(oscillations)

    def _compute_improvement_ratio(self, vals: List[float]) -> Optional[float]:
        """Compute fraction of rounds with improvement."""
        if len(vals) < 2:
            return None

        best_so_far = vals[0]
        improvements = 0
        for val in vals[1:]:
            if val < best_so_far:
                improvements += 1
                best_so_far = val

        return float(improvements) / float(len(vals) - 1)

    def _compute_improvement_magnitude(self, vals: List[float]) -> Optional[float]:
        """Compute average magnitude of loss reduction between improvements."""
        if len(vals) < 2:
            return None

        # Extract loss values at improvement points
        last_impr_loss: List[float] = []
        best_so_far = vals[0]
        for val in vals:
            if val < best_so_far:
                last_impr_loss.append(val)
                best_so_far = val

        # Calculate deltas between consecutive improvement loss values
        run_deltas = []
        for i in range(1, len(last_impr_loss)):
            delta = last_impr_loss[i - 1] - last_impr_loss[i]
            if delta > 0:
                run_deltas.append(delta)

        return float(np.mean(run_deltas)) if run_deltas else None

    # =========================================================================
    # Loading data
    # =========================================================================

    def load_configs(self, experiment_name: str) -> Dict:
        """
        Load the configuration for an experiment.

        Args:
            experiment_name: Name of the experiment.

        Returns:
            Dictionary containing the configuration, or empty dict if not found.
        """
        cfg = self._read_json(self._config_path(experiment_name))
        return dict(cfg) if isinstance(cfg, dict) else {}

    def load_runs_stats(self, experiment_name: str) -> Dict[str, Dict[str, float]]:
        """
        Load and aggregate statistics for all runs of an experiment.

        For each metric column in the server CSV files, computes:
        - Per-run aggregate (based on agg_mode)
        - Mean and std across all runs

        Also computes bandwidth statistics (downlink, uplink, total) and stability metrics
        (derived from the chosen loss time-series).

        Time values are converted from seconds to the configured time_unit.
        Size values are converted from MB to the configured size_unit.

        Args:
            experiment_name: Name of the experiment.

        Returns:
            Dictionary with 'aggregates' key containing all computed statistics.
        """
        base = self.runs_dir / experiment_name
        if not base.exists():
            logger.warning("Experiment folder not found: %s", base)
            return {"aggregates": {}}

        # load experiment config to resolve 'loss' -> personal/global
        cfg = self.load_configs(experiment_name)
        loss_metric_name = self.resolve_loss_metric(cfg.get("save_local_model", False))

        runs: List[Dict] = []
        numeric_cols: set[str] = set()
        bandwidth_per_run: Dict[str, List[float]] = {
            "downlink": [],
            "uplink": [],
            "total": [],
        }

        for entry in sorted(base.iterdir()):
            if not entry.is_dir():
                continue

            data = self._read_server_csv(entry)
            if data is None:
                logger.debug("Skipping run (no server.csv): %s", entry)
                continue

            for k, v in data.items():
                if isinstance(v, list) and k != "send_mb":
                    numeric_cols.add(k)

            runs.append({"name": entry.name, **data})

            bw = self._compute_bandwidth_stats(entry)
            for key in ["downlink", "uplink", "total"]:
                agg_val = self._compute_per_run_agg(bw[key])
                if agg_val is not None:
                    bandwidth_per_run[key].append(agg_val)

        aggregates: Dict[str, float] = {}

        for col in sorted(numeric_cols):
            per_run_agg: List[float] = []
            for r in runs:
                seq = r.get(col, [])
                vals = self._parse_numeric_list(seq, exclude_zero=False)
                agg_val = self._compute_per_run_agg(vals)
                if agg_val is not None:
                    per_run_agg.append(agg_val)

            # Rename time_per_iter to efficiency
            agg_col = "efficiency" if col == "time_per_iter" else col
            is_time = self._is_time_column(agg_col)
            aggregates.update(
                self._compute_aggregates(per_run_agg, agg_col, is_time=is_time)
            )

        for bw_key in ["downlink", "uplink", "total"]:
            # Rename total to communication
            agg_key = "communication" if bw_key == "total" else bw_key
            aggregates.update(
                self._compute_aggregates(
                    bandwidth_per_run[bw_key], agg_key, is_size=True
                )
            )

        # =====================================================================
        # Per-run loss-derived metrics
        # =====================================================================
        run_vals = [
            (
                r,
                self._parse_numeric_list(
                    r.get(loss_metric_name, []), exclude_zero=False
                ),
            )
            for r in runs
        ]
        valid_run_vals = [(r, vals) for r, vals in run_vals if len(vals) >= 2]

        # Aggregate all metrics
        aggregates.update(
            self._compute_aggregates(
                [
                    v
                    for r, vals in valid_run_vals
                    if (v := self._compute_last_improvement_round(vals)) is not None
                ],
                "last_improvement_round",
            )
        )

        aggregates.update(
            self._compute_aggregates(
                [
                    v
                    for r, vals in valid_run_vals
                    if (v := self._compute_improvement_streaks(vals)[0]) is not None
                ],
                "longest_improvement_streak",
            )
        )

        aggregates.update(
            self._compute_aggregates(
                [
                    v
                    for r, vals in valid_run_vals
                    if (v := self._compute_improvement_streaks(vals)[1]) is not None
                ],
                "most_frequent_improvement_streak",
            )
        )

        aggregates.update(
            self._compute_aggregates(
                [
                    v
                    for r, vals in valid_run_vals
                    if (v := self._compute_oscillation_count(vals)) is not None
                ],
                "oscillation_count",
            )
        )

        aggregates.update(
            self._compute_aggregates(
                [
                    v
                    for r, vals in valid_run_vals
                    if (v := self._compute_improvement_ratio(vals)) is not None
                ],
                "improvement_ratio",
            )
        )

        aggregates.update(
            self._compute_aggregates(
                [
                    v
                    for r, vals in valid_run_vals
                    if (v := self._compute_improvement_magnitude(vals)) is not None
                ],
                "improvement_magnitude",
            )
        )

        return {"aggregates": aggregates}

    # =========================================================================
    # Filtering
    # =========================================================================

    def _matches_filter(
        self,
        experiment: Dict,
        models: Optional[List[str]] = None,
        strategies: Optional[List[str]] = None,
        datasets: Optional[List[str]] = None,
        output_lens: Optional[List[int]] = None,
        experiments: Optional[List[str]] = None,
    ) -> bool:
        """
        Check if an experiment matches all provided filters.

        Args:
            experiment: Experiment metadata dictionary.
            models: List of model names to include (case-insensitive substring matching).
            strategies: List of strategy names to include (case-insensitive substring matching).
            datasets: List of dataset names to include (case-insensitive substring matching).
            output_lens: List of output lengths to include.
            experiments: List of experiment name patterns to include (case-insensitive substring matching).

        Returns:
            True if the experiment matches all non-None filters.
        """
        exp_name = experiment.get("exp", "").lower()

        if experiments is not None:
            if not any(e.lower() in exp_name for e in experiments):
                return False

        if models is not None:
            exp_model = experiment.get("model", "").lower()
            if not any(m.lower() in exp_model for m in models):
                return False

        if strategies is not None:
            exp_strategy = experiment.get("strategy", "").lower()
            if not any(s.lower() in exp_strategy for s in strategies):
                return False

        if datasets is not None:
            exp_dataset = experiment.get("dataset", "").lower()
            if not any(d.lower() in exp_dataset for d in datasets):
                return False

        if output_lens is not None:
            exp_output_len = experiment.get("output_len")
            if exp_output_len is None or int(exp_output_len) not in output_lens:
                return False

        return True

    def load_all_experiments(
        self,
        models: Optional[List[str]] = None,
        strategies: Optional[List[str]] = None,
        datasets: Optional[List[str]] = None,
        output_lens: Optional[List[int]] = None,
        experiments: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Load all experiments from the runs directory with optional filtering.

        Args:
            models: Filter to specific models.
            strategies: Filter to specific strategies.
            datasets: Filter to specific datasets.
            output_lens: Filter to specific output lengths.
            experiments: Filter to specific experiment name patterns.

        Returns:
            List of experiment dictionaries containing config and aggregated stats.
        """
        all_experiments: List[Dict] = []

        if not self.runs_dir.exists():
            logger.warning("Runs directory '%s' not found", self.runs_dir)
            return all_experiments

        for child in sorted(self.runs_dir.iterdir()):
            if not child.is_dir():
                continue

            if not self._results_path(child.name).exists():
                logger.debug("Skipping experiment (no results.csv): %s", child)
                continue

            datum: Dict = {"exp": child.name}
            datum.update(self.load_configs(child.name))

            if not self._matches_filter(
                experiment=datum,
                models=models,
                strategies=strategies,
                datasets=datasets,
                output_lens=output_lens,
                experiments=experiments,
            ):
                logger.debug("Skipping experiment (filtered out): %s", child.name)
                continue

            stats = self.load_runs_stats(child.name)
            datum.update(stats)
            all_experiments.append(datum)

        return all_experiments

    # =========================================================================
    # Display methods
    # =========================================================================

    def _resolve_metric(self, metric: str, experiment: Dict) -> str:
        """
        Resolve metric name, handling special metrics.

        Args:
            metric: Metric name (can be 'loss' or actual metric name).
            experiment: Experiment dictionary with config.

        Returns:
            Resolved metric name.
        """
        if metric == "loss":
            save_local_model = experiment.get("save_local_model", False)
            return self.resolve_loss_metric(save_local_model)
        return metric

    def _get_metric_value_str(self, experiment: Dict, metric: str) -> Optional[str]:
        """
        Get formatted mean±std string for a metric from an experiment.

        Args:
            experiment: Experiment dictionary with aggregates.
            metric: Base metric name (e.g., 'loss', 'global_avg_test_loss').

        Returns:
            Formatted string "mean±std" or None if not available.
        """
        resolved_metric = self._resolve_metric(metric, experiment)
        aggregates = experiment.get("aggregates", {})
        mean_key = f"{resolved_metric}_{self.agg_mode}_mean"
        std_key = f"{resolved_metric}_{self.agg_mode}_std"

        mean_val = aggregates.get(mean_key)
        std_val = aggregates.get(std_key)

        if mean_val is None or std_val is None:
            return None
        if mean_val == self.output_sentinel and std_val == self.output_sentinel:
            return None

        return self._format_mean_std_str(mean_val, std_val)

    def _get_pivot_config(self, pivot: PivotOption) -> Tuple[str, str, Tuple[str, ...]]:
        """
        Get configuration for pivot mode.

        Args:
            pivot: Pivot option ('model' or 'strategy').

        Returns:
            Tuple of (group_by_field, pivot_by_field, group_by_tuple).
            - group_by_field: Field to group tables by (iterate over unique values)
            - pivot_by_field: Field to pivot columns by
            - group_by_tuple: Fields for row grouping
        """
        if pivot == "model":
            # Group tables by model, pivot columns by strategy
            return "model", "strategy", ("dataset", "input_len", "output_len")
        else:  # strategy
            # Group tables by strategy, pivot columns by model
            return "strategy", "model", ("dataset", "input_len", "output_len")

    def create_table(
        self,
        experiments: List[Dict],
        metric: str = "loss",
        group_by: Tuple[str, ...] = ("dataset", "input_len", "output_len"),
        pivot_by: str = "strategy",
    ) -> pl.DataFrame:
        """
        Create a pivot table displaying metric values grouped by configuration.

        Args:
            experiments: List of experiment dictionaries.
            metric: Metric to display (e.g., 'loss', 'downlink').
            group_by: Tuple of fields to group rows by.
            pivot_by: Field to pivot columns by.

        Returns:
            Polars DataFrame with the pivot table.
        """
        # Build rows
        rows: List[Dict[str, Any]] = []
        for exp in experiments:
            row: Dict[str, Any] = {}

            # Add group_by fields
            for field in group_by:
                row[field] = exp.get(field)

            # Add pivot field
            row[pivot_by] = exp.get(pivot_by)

            # Add metric value
            row["value"] = self._get_metric_value_str(exp, metric)

            rows.append(row)

        if not rows:
            logger.warning("No data to display")
            return pl.DataFrame()

        # Create DataFrame
        df = pl.DataFrame(rows)

        # Get unique pivot values for column ordering
        pivot_values = df.select(pivot_by).unique().sort(pivot_by).to_series().to_list()

        # Pivot the table
        df_pivot = df.pivot(
            on=pivot_by,
            index=list(group_by),
            values="value",
            aggregate_function="first",
        )

        # Reorder columns: group_by fields first, then pivot values in sorted order
        col_order = list(group_by) + [
            col for col in pivot_values if col in df_pivot.columns
        ]
        df_pivot = df_pivot.select(col_order)

        # Sort by group_by fields
        df_pivot = df_pivot.sort(list(group_by))

        return df_pivot

    def _get_table_header(
        self,
        group_value: str,
        metric: str,
        pivot: PivotOption,
    ) -> str:
        """
        Generate header text for a table.

        Args:
            group_value: Value of the grouping field (model or strategy name).
            metric: Metric name.
            pivot: Pivot mode ('model' or 'strategy').

        Returns:
            Formatted header string.
        """
        # Resolve metric for display
        resolved_metric = metric

        if pivot == "model":
            group_label = "MODEL"
            pivot_label = "strategy"
        else:
            group_label = "STRATEGY"
            pivot_label = "model"

        lines = [
            "=" * 80,
            f"{group_label}: {group_value.upper()}",
            f"METRIC: {metric}",
        ]

        # Add metric description if available
        if metric in METRIC_DESCRIPTIONS:
            desc = METRIC_DESCRIPTIONS[metric]
            lines.append(f"  {desc['title']}")
            lines.append("")
            for line in desc["explanation"].split("\n"):
                if line.strip():
                    lines.append(f"  {line}")
            lines.append("")

        lines.extend(
            [
                f"(Standard deviation multiplied by {self.std_multiplier})",
                f"(Displayed with {self.decimal_places} decimal places)",
                f"(Aggregation mode: {self.agg_mode})",
            ]
        )
        if resolved_metric in TIME_COLUMNS:
            lines.append(f"(Time unit: {self.time_unit})")
        if resolved_metric in SIZE_COLUMNS:
            lines.append(f"(Size unit: {self.size_unit})")
        lines.extend(
            [
                "=" * 80,
                "Each row shows a unique configuration (dataset, in, out)",
                f"Columns show mean±std across multiple runs for each {pivot_label}",
                "-" * 80,
            ]
        )
        return "\n".join(lines)

    def _get_ranking_table_header(
        self,
        group_value: str,
        metric: str,
        pivot: PivotOption,
    ) -> str:
        """
        Generate header text for a ranking table.

        Args:
            group_value: Value of the grouping field (model or strategy name).
            metric: Metric name.
            pivot: Pivot mode ('model' or 'strategy').

        Returns:
            Formatted header string.
        """
        if pivot == "model":
            group_label = "MODEL"
            pivot_label = "strategy"
            best_label = "best_strategy"
        else:
            group_label = "STRATEGY"
            pivot_label = "model"
            best_label = "best_model"

        lines = [
            "=" * 80,
            f"RANKING TABLE FOR {group_label}: {group_value.upper()}",
            f"METRIC: {metric}",
        ]

        # Add metric description if available
        if metric in METRIC_DESCRIPTIONS:
            desc = METRIC_DESCRIPTIONS[metric]
            lines.append(f"  {desc['title']}")
            lines.append("")
            for line in desc["explanation"].split("\n"):
                if line.strip():
                    lines.append(f"  {line}")
            lines.append("")

        lines.extend(
            [
                "=" * 80,
                f"{pivot_label.capitalize()}s ranked by mean performance (1=best, lower is better)",
                "Ties broken by standard deviation (lower std is better)",
                "Last row shows average rank across all configurations",
                f"Last column shows which {pivot_label} wins most often",
                "-" * 80,
            ]
        )
        return "\n".join(lines)

    def _get_metric_mean_std(
        self, experiment: Dict, metric: str
    ) -> Optional[Tuple[float, float]]:
        """
        Get mean and std values for a metric from an experiment.

        Args:
            experiment: Experiment dictionary with aggregates.
            metric: Base metric name.

        Returns:
            Tuple of (mean, std) or None if not available.
        """
        resolved_metric = self._resolve_metric(metric, experiment)
        aggregates = experiment.get("aggregates", {})
        mean_key = f"{resolved_metric}_{self.agg_mode}_mean"
        std_key = f"{resolved_metric}_{self.agg_mode}_std"

        mean_val = aggregates.get(mean_key)
        std_val = aggregates.get(std_key)

        if mean_val is None or std_val is None:
            return None
        if mean_val == self.output_sentinel and std_val == self.output_sentinel:
            return None

        return (mean_val, std_val)

    def create_ranking_table(
        self,
        experiments: List[Dict],
        metric: str = "loss",
        group_by: Tuple[str, ...] = ("dataset", "input_len", "output_len"),
        pivot_by: str = "strategy",
        lower_is_better: bool = True,
    ) -> pl.DataFrame:
        """
        Create a ranking table displaying strategy rankings grouped by configuration.

        Ranking rules:
        - Strategies are ranked by mean performance (1=best)
        - When means are equal, lower standard deviation wins
        - N/A for strategies with no data for a configuration

        Args:
            experiments: List of experiment dictionaries.
            metric: Metric to rank by.
            group_by: Tuple of fields to group rows by.
            pivot_by: Field to pivot columns by (strategies or models).
            lower_is_better: If True, lower metric values get better ranks.

        Returns:
            Polars DataFrame with ranking table including:
            - Pivot columns with ranks (1, 2, 3, ... or N/A)
            - best_{pivot_by} column showing winner for each row
            - AVG_RANK row showing average ranks
            - Most frequent winner in bottom-right cell
        """
        # Build data structure: {config_key: {pivot_value: (mean, std)}}
        config_data: Dict[Tuple, Dict[str, Tuple[float, float]]] = {}
        all_pivot_values: set[str] = set()

        for exp in experiments:
            # Build config key
            config_key = tuple(exp.get(field) for field in group_by)
            pivot_value = exp.get(pivot_by)

            if pivot_value is None:
                continue

            all_pivot_values.add(pivot_value)

            # Get mean and std
            mean_std = self._get_metric_mean_std(exp, metric)
            if mean_std is None:
                continue

            if config_key not in config_data:
                config_data[config_key] = {}
            config_data[config_key][pivot_value] = mean_std

        if not config_data:
            logger.warning("No data to rank")
            return pl.DataFrame()

        # Sort pivot values for consistent column ordering
        pivot_values = sorted(all_pivot_values)

        # Best column name
        best_col = f"best_{pivot_by}"

        # Compute ranks for each configuration
        rows: List[Dict[str, Any]] = []
        pivot_ranks: Dict[str, List[float]] = {pv: [] for pv in pivot_values}
        win_counts: Dict[str, int] = {pv: 0 for pv in pivot_values}

        for config_key in sorted(config_data.keys()):
            row: Dict[str, Any] = {}

            # Add group_by fields
            for i, field in enumerate(group_by):
                row[field] = config_key[i]

            # Get pivot values with data for this config
            config_pivot_values = config_data[config_key]

            # Sort pivot values by (mean, std) - lower is better if lower_is_better
            sorted_pivot_values = sorted(
                config_pivot_values.items(),
                key=lambda x: (x[1][0], x[1][1]),
                reverse=not lower_is_better,
            )

            # Assign ranks (handle ties)
            ranks: Dict[str, int] = {}
            current_rank = 1
            prev_mean_std: Optional[Tuple[float, float]] = None

            for i, (pv, mean_std) in enumerate(sorted_pivot_values):
                if prev_mean_std is not None and mean_std == prev_mean_std:
                    # Tie - same rank as previous
                    ranks[pv] = ranks[sorted_pivot_values[i - 1][0]]
                else:
                    ranks[pv] = current_rank
                prev_mean_std = mean_std
                current_rank = i + 2  # Next rank after this position

            # Fill in ranks for all pivot values
            best_pivot: Optional[str] = None
            best_rank = float("inf")

            for pv in pivot_values:
                if pv in ranks:
                    rank = ranks[pv]
                    row[pv] = str(rank)
                    pivot_ranks[pv].append(float(rank))
                    if rank < best_rank:
                        best_rank = rank
                        best_pivot = pv
                else:
                    row[pv] = "N/A"

            # Track winner
            if best_pivot:
                win_counts[best_pivot] += 1
                row[best_col] = best_pivot
            else:
                row[best_col] = "N/A"

            rows.append(row)

        # Compute average ranks
        avg_rank_row: Dict[str, Any] = {}
        for i, field in enumerate(group_by):
            if i == 0:
                avg_rank_row[field] = "AVG_RANK"
            else:
                avg_rank_row[field] = ""

        for pv in pivot_values:
            if pivot_ranks[pv]:
                avg = np.mean(pivot_ranks[pv])
                avg_rank_row[pv] = f"{avg:.2f}"
            else:
                avg_rank_row[pv] = "N/A"

        # Determine most frequent winner with tiebreaking
        max_wins = max(win_counts.values()) if win_counts else 0
        winners = [
            pv for pv, count in win_counts.items() if count == max_wins and count > 0
        ]

        if len(winners) == 1:
            most_frequent = f"{winners[0]} ({max_wins}x)"
        elif len(winners) > 1:
            # Tiebreak by average rank (lower is better)
            winner_avg_ranks = []
            for w in winners:
                if pivot_ranks[w]:
                    winner_avg_ranks.append((w, np.mean(pivot_ranks[w])))
                else:
                    winner_avg_ranks.append((w, float("inf")))
            winner_avg_ranks.sort(key=lambda x: x[1])

            # Check if still tied after avg rank
            best_avg = winner_avg_ranks[0][1]
            final_winners = [w for w, avg in winner_avg_ranks if avg == best_avg]

            if len(final_winners) == 1:
                most_frequent = f"{final_winners[0]} ({max_wins}x)"
            else:
                # Still tied - show all
                tie_str = ", ".join(f"{w} ({max_wins}x)" for w in final_winners)
                most_frequent = f"tie: {tie_str}"
        else:
            most_frequent = "N/A"

        avg_rank_row[best_col] = most_frequent
        rows.append(avg_rank_row)

        # Create DataFrame
        df = pl.DataFrame(rows)

        # Reorder columns
        col_order = list(group_by) + pivot_values + [best_col]
        df = df.select([c for c in col_order if c in df.columns])

        return df

    def save_tables(
        self,
        experiments: List[Dict],
        metric: str = "loss",
        pivot: PivotOption = "model",
        include_ranking: bool = True,
        lower_is_better: bool = True,
    ) -> None:
        """
        Save tables grouped by the specified pivot field to the output directory.

        Creates one CSV file per unique value of the grouping field.
        Optionally also creates ranking tables.
        Also prints the tables to stdout.

        Args:
            experiments: List of experiment dictionaries.
            metric: Metric to display.
            pivot: Pivot mode - 'model' groups by model and pivots by strategy,
                   'strategy' groups by strategy and pivots by model.
            include_ranking: If True, also generate ranking tables.
            lower_is_better: If True, lower metric values get better ranks (for ranking).
        """
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get pivot configuration
        group_by_field, pivot_by_field, row_group_by = self._get_pivot_config(pivot)

        # Get unique values of the grouping field
        group_values = sorted(
            set(exp.get(group_by_field, "unknown") for exp in experiments)
        )

        for group_value in group_values:
            # Filter experiments for this group
            group_exps = [
                exp for exp in experiments if exp.get(group_by_field) == group_value
            ]

            if not group_exps:
                continue

            # Create and save value table
            df = self.create_table(
                experiments=group_exps,
                metric=metric,
                group_by=row_group_by,
                pivot_by=pivot_by_field,
            )

            if df.is_empty():
                logger.warning("No data for %s: %s", group_by_field, group_value)
                continue

            # Print header and table
            header = self._get_table_header(group_value, metric, pivot)
            print(header)
            print(df)
            print()

            # Save to CSV
            filename = f"{group_value}_{metric}_{self.agg_mode}.csv"
            output_path = self.output_dir / filename
            df.write_csv(output_path)
            logger.info("Saved table to: %s", output_path)

            # Create and save ranking table if requested
            if include_ranking:
                df_rank = self.create_ranking_table(
                    experiments=group_exps,
                    metric=metric,
                    group_by=row_group_by,
                    pivot_by=pivot_by_field,
                    lower_is_better=lower_is_better,
                )

                if not df_rank.is_empty():
                    # Print ranking table
                    rank_header = self._get_ranking_table_header(
                        group_value, metric, pivot
                    )
                    print(rank_header)
                    print(df_rank)
                    print()

                    # Save ranking table
                    rank_filename = (
                        f"{group_value}_{metric}_{self.agg_mode}_ranking.csv"
                    )
                    rank_output_path = self.output_dir / rank_filename
                    df_rank.write_csv(rank_output_path)
                    logger.info("Saved ranking table to: %s", rank_output_path)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze and aggregate federated learning experiment results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--runs-dir",
        "-r",
        type=str,
        default="runs",
        help="Path to the runs directory",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="analysis/tables",
        help="Output directory for generated tables",
    )
    parser.add_argument(
        "--std-multiplier",
        "-s",
        type=float,
        default=1e3,
        help="Factor to multiply standard deviation values",
    )
    parser.add_argument(
        "--decimal-places",
        "-d",
        type=int,
        default=3,
        help="Number of decimal places to round output values",
    )
    parser.add_argument(
        "--agg-mode",
        "-a",
        type=str,
        choices=AGG_MODES,
        default="min",
        help="Per-run aggregation mode",
    )
    parser.add_argument(
        "--time-unit",
        "-t",
        type=str,
        choices=TIME_UNITS,
        default="s",
        help="Output unit for time values (s=seconds, ms=milliseconds, m=minutes, h=hours)",
    )
    parser.add_argument(
        "--size-unit",
        "-z",
        type=str,
        choices=SIZE_UNITS,
        default="mb",
        help="Output unit for size values (b=bytes, kb=kilobytes, mb=megabytes, gb=gigabytes, tb=terabytes)",
    )
    parser.add_argument(
        "--metric",
        "-m",
        type=str,
        choices=METRICS,
        default="loss",
        help="Metric to display. Use 'efficiency' for time per iteration, 'communication' for total bandwidth. "
        "'loss' resolves to personal_avg_test_loss or global_avg_test_loss based on save_local_model. "
        "Use 'all' to generate tables for all available metrics.",
    )
    parser.add_argument(
        "--pivot",
        "-p",
        type=str,
        choices=PIVOT_OPTIONS,
        default="model",
        help="Pivot mode: 'model' groups by model with strategy columns, "
        "'strategy' groups by strategy with model columns",
    )
    parser.add_argument(
        "--no-ranking",
        action="store_true",
        help="Disable ranking table generation",
    )
    parser.add_argument(
        "--higher-is-better",
        action="store_true",
        help="Higher metric values are better (default: lower is better)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--excels",
        "-e",
        type=str,
        nargs="+",
        help="Excel files with 'project' and 'name' columns for batch processing",
    )

    # Filtering options
    filter_group = parser.add_argument_group("filtering options")
    filter_group.add_argument(
        "--models",
        type=str,
        nargs="+",
        metavar="MODEL",
        help="Filter to specific models",
    )
    filter_group.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        metavar="STRATEGY",
        help="Filter to specific strategies",
    )
    filter_group.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        metavar="DATASET",
        help="Filter to specific datasets",
    )
    filter_group.add_argument(
        "--output-lens",
        type=int,
        nargs="+",
        metavar="LEN",
        help="Filter to specific output lengths",
    )
    filter_group.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        metavar="NAME",
        help="Filter to specific experiment name patterns",
    )

    return parser.parse_args()


def _load_queries_from_excels(
    excels: List[str],
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Load (project, name, row) tuples from Excel files."""
    queries: List[Tuple[str, str, Dict[str, Any]]] = []
    for excel_file in excels:
        try:
            df = pl.read_excel(excel_file)
            for row in df.iter_rows(named=True):
                project = row.get("--project=")
                name = row.get("--name=")
                if project and name:
                    queries.append((project, name, row))
                else:
                    logger.warning(
                        "Skipping row with missing --project= or --name= in %s: %s",
                        excel_file,
                        row,
                    )
            logger.info("Loaded queries from Excel file: %s", excel_file)
        except Exception as e:
            logger.error("Failed to read Excel file %s: %s", excel_file, e)
    return queries


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine queries to process
    if args.excels:
        queries = _load_queries_from_excels(args.excels)
        if not queries:
            logger.error("No valid queries loaded from Excel files")
            return
        logger.info("Total queries to process: %d", len(queries))
    else:
        # Use command-line arguments as single query
        queries = [(args.runs_dir, None, None)]

    # Load all experiments from all queries and track missing ones
    all_experiments = []
    missing_experiments = []

    for query_item in queries:
        runs_dir, exp_name, row = query_item

        analysis = Analysis(
            runs_dir=runs_dir,
            output_dir=args.output_dir,
            std_multiplier=args.std_multiplier,
            decimal_places=args.decimal_places,
            agg_mode=args.agg_mode,
            time_unit=args.time_unit,
            size_unit=args.size_unit,
        )

        # Load only the specific experiment from Excel, or all experiments if using CLI
        if exp_name:
            # From Excel: load only the specified experiment
            experiments = analysis.load_all_experiments(
                experiments=[exp_name],
                models=args.models,
                strategies=args.strategies,
                datasets=args.datasets,
                output_lens=args.output_lens,
            )
            if not experiments and row:
                # Experiment not found, store for later logging
                missing_experiments.append((exp_name, row))
        else:
            # From CLI: load all experiments with optional filters
            experiments = analysis.load_all_experiments(
                models=args.models,
                strategies=args.strategies,
                datasets=args.datasets,
                output_lens=args.output_lens,
                experiments=args.experiments,
            )

        all_experiments.extend(experiments)

    # Log missing experiments with their script commands
    if missing_experiments:
        logger.warning("\n" + "=" * 80)
        logger.warning("MISSING EXPERIMENTS - Run these commands to generate results:")
        logger.warning("=" * 80)
        for exp_name, row in missing_experiments:
            script = row.get("script", "N/A") if row else "N/A"
            print(script)
        logger.warning("\n" + "=" * 80)

    if not all_experiments:
        logger.warning("No experiments loaded from any query")
        return

    logger.info("Total experiments loaded: %d", len(all_experiments))

    # Determine which metrics to process
    if args.metric == "all":
        metrics_to_process = [m for m in METRICS if m != "all"]
        logger.info("Processing all metrics: %s", ", ".join(metrics_to_process))
    else:
        metrics_to_process = [args.metric]

    # Create Analysis instance for save_tables (uses first runs_dir for output_dir reference)
    analysis = Analysis(
        runs_dir=queries[0][0],
        output_dir=args.output_dir,
        std_multiplier=args.std_multiplier,
        decimal_places=args.decimal_places,
        agg_mode=args.agg_mode,
        time_unit=args.time_unit,
        size_unit=args.size_unit,
    )

    # Process each metric once with all experiments combined
    for metric in metrics_to_process:
        if len(metrics_to_process) > 1:
            logger.info("Processing metric: %s", metric)

        analysis.save_tables(
            all_experiments,
            metric=metric,
            pivot=args.pivot,
            include_ranking=not args.no_ranking,
            lower_is_better=not args.higher_is_better,
        )


if __name__ == "__main__":
    main()
