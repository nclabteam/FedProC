"""
Per-experiment analysis.

Reads all runs of one experiment, computes per-run aggregates,
aggregates across runs (mean±std), and saves results.csv.

Input:  runs/expN/ directory
Output: runs/expN/results.csv (one row per metric, avg_min/std_min/avg_max/std_max)
"""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl

sys.path.append(str(Path(__file__).resolve().parent.parent))

from analysis.io import (
    DEFAULT_VALUE,
    load_config,
    parse_numeric_list,
    read_csv,
    read_timing,
    resolve_loss_metric,
)
from analysis.metrics import (
    compute_per_run_agg,
    improvement_magnitude,
    improvement_ratio,
    improvement_streaks,
    last_improvement_round,
    oscillation_count,
)

METRIC_UNITS = {
    "efficiency": "s",
    "communication": "MB",
    "uplink": "MB",
    "downlink": "MB",
    "time_per_experiment": "s",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ExperimentAnalysis:
    """Analyze all runs of a single experiment."""

    def __init__(
        self,
        experiment_dir: str | Path,
        agg_mode: str = "min",
        decimal_places: int = 4,
    ) -> None:
        self.experiment_dir = Path(experiment_dir)
        self.agg_mode = agg_mode
        self.decimal_places = decimal_places

    def _read_server_csv(self, run_dir: Path) -> Optional[Dict[str, List]]:
        """Read server.csv from a run directory (or compacted root)."""
        # Try standard layout: run_dir/results/server.csv
        p = run_dir / "results" / "server.csv"
        if p.exists():
            return read_csv(p)
        # Compacted layout: experiment_dir/server.csv
        p = run_dir / "server.csv"
        if p.exists():
            return read_csv(p)
        return None

    def _read_client_csvs(self, run_dir: Path) -> List[Dict[str, List]]:
        """Read all client CSVs from a run directory (or compacted root)."""
        # Try standard layout: run_dir/results/client_*.csv
        results_dir = run_dir / "results"
        if results_dir.exists():
            clients = []
            for f in sorted(results_dir.glob("client_*.csv")):
                data = read_csv(f)
                if data is not None:
                    clients.append(data)
            if clients:
                return clients
        # Compacted layout: experiment_dir/client.csv (single merged file)
        compact_client = run_dir / "client.csv"
        if compact_client.exists():
            data = read_csv(compact_client)
            if data is not None:
                return [data]
        return []

    def _compute_bandwidth_stats(self, run_dir: Path) -> Dict[str, List[float]]:
        """Compute per-round bandwidth stats for a single run."""
        server_data = self._read_server_csv(run_dir)
        client_datas = self._read_client_csvs(run_dir)

        downlink_vals = []
        if server_data and "downlink_mb" in server_data:
            downlink_vals = parse_numeric_list(
                server_data["downlink_mb"], exclude_zero=True
            )

        num_rounds = 0
        if server_data and "downlink_mb" in server_data:
            num_rounds = len(server_data["downlink_mb"])
        for cd in client_datas:
            if "uplink_mb" in cd:
                num_rounds = max(num_rounds, len(cd["uplink_mb"]))

        # Uplink: sum of all clients' uplink_mb per round
        uplink_per_round = []
        for i in range(num_rounds):
            round_sum = 0.0
            has_valid = False
            for cd in client_datas:
                if "uplink_mb" not in cd or i >= len(cd["uplink_mb"]):
                    continue
                try:
                    xv = float(cd["uplink_mb"][i])
                    if xv != DEFAULT_VALUE:
                        round_sum += xv
                        has_valid = True
                except (ValueError, TypeError):
                    continue
            if has_valid and round_sum > 0:
                uplink_per_round.append(round_sum)

        # Total: downlink + uplink per round
        total_per_round = []
        if server_data and "downlink_mb" in server_data:
            for i in range(num_rounds):
                downlink_i = None
                if i < len(server_data["downlink_mb"]):
                    try:
                        dv = float(server_data["downlink_mb"][i])
                        if dv != DEFAULT_VALUE and dv > 0:
                            downlink_i = dv
                    except (ValueError, TypeError):
                        pass
                uplink_i = 0.0
                uplink_valid = False
                for cd in client_datas:
                    if "uplink_mb" not in cd or i >= len(cd["uplink_mb"]):
                        continue
                    try:
                        uv = float(cd["uplink_mb"][i])
                        if uv != DEFAULT_VALUE:
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

    def load_run(self, run_dir: Path) -> Dict[str, float]:
        """Load one run and compute per-run aggregate for each metric."""
        data = self._read_server_csv(run_dir)
        if data is None:
            return {}

        result = {}
        skip_cols = {"round"}
        for col, values in data.items():
            if not isinstance(values, list) or col in skip_cols:
                continue
            vals = parse_numeric_list(values)
            agg_val = compute_per_run_agg(vals, self.agg_mode)
            if agg_val is not None:
                # Rename time_per_iter to efficiency
                key = "efficiency" if col == "time_per_iter" else col
                result[key] = agg_val

        # Bandwidth stats
        bw = self._compute_bandwidth_stats(run_dir)
        for bw_key, bw_vals in bw.items():
            agg_val = compute_per_run_agg(bw_vals, self.agg_mode)
            if agg_val is not None:
                # Rename total to communication
                key = "communication" if bw_key == "total" else bw_key
                result[key] = agg_val

        return result

    def load_all_runs(self) -> List[Dict[str, float]]:
        """Load all runs and compute per-run aggregates."""
        runs = []
        for entry in sorted(self.experiment_dir.iterdir()):
            if not entry.is_dir() or not entry.name.isdigit():
                continue
            run_data = self.load_run(entry)
            if run_data:
                runs.append(run_data)
        # Fallback: compacted experiment (server.csv directly in root)
        if not runs:
            run_data = self.load_run(self.experiment_dir)
            if run_data:
                runs.append(run_data)
        return runs

    def compute_derived_metrics(
        self, runs_data: List[Dict[str, List[float]]]
    ) -> Dict[str, float]:
        """Compute derived metrics from per-run loss time-series."""
        cfg = load_config(self.experiment_dir)
        loss_metric = resolve_loss_metric()

        derived = defaultdict(list)
        for run in runs_data:
            loss_series = parse_numeric_list(run.get(loss_metric, []))
            if len(loss_series) < 2:
                continue

            if (v := last_improvement_round(loss_series)) is not None:
                derived["last_improvement_round"].append(v)

            longest, most_freq = improvement_streaks(loss_series)
            if longest is not None:
                derived["longest_improvement_streak"].append(longest)
            if most_freq is not None:
                derived["most_frequent_improvement_streak"].append(most_freq)

            if (v := oscillation_count(loss_series)) is not None:
                derived["oscillation_count"].append(v)

            if (v := improvement_ratio(loss_series)) is not None:
                derived["improvement_ratio"].append(v)

            if (v := improvement_magnitude(loss_series)) is not None:
                derived["improvement_magnitude"].append(v)

        return dict(derived)

    def analyze(self) -> pl.DataFrame:
        """Full pipeline: load runs → aggregate → derive → return results."""
        # Load per-run data (raw time-series for derived metrics)
        runs_raw = []
        for entry in sorted(self.experiment_dir.iterdir()):
            if not entry.is_dir() or not entry.name.isdigit():
                continue
            server_data = self._read_server_csv(entry)
            if server_data is not None:
                runs_raw.append(server_data)
        # Fallback: compacted experiment
        if not runs_raw:
            server_data = self._read_server_csv(self.experiment_dir)
            if server_data is not None:
                runs_raw.append(server_data)

        # Load per-run aggregates
        runs_agg = self.load_all_runs()
        if not runs_agg:
            logger.warning("No valid runs found in %s", self.experiment_dir)
            return pl.DataFrame()

        # Aggregate across runs
        stats = defaultdict(lambda: {"min": [], "max": [], "avg": []})
        for run in runs_agg:
            for key, value in run.items():
                stats[key]["min"].append(value)
                stats[key]["max"].append(value)

        runs_agg_mean = ExperimentAnalysis(
            self.experiment_dir, agg_mode="mean", decimal_places=self.decimal_places
        ).load_all_runs()
        for run in runs_agg_mean:
            for key, value in run.items():
                stats[key]["avg"].append(value)

        # Add time per experiment from timing.json
        timing_path = self.experiment_dir / "timing.json"
        timing_entries = read_timing(timing_path)
        if timing_entries:
            durations = [e["seconds"] for e in timing_entries if "seconds" in e]
            if durations:
                stats["time_per_experiment"]["min"] = durations
                stats["time_per_experiment"]["max"] = durations
                stats["time_per_experiment"]["avg"] = durations

        # Compute derived metrics
        derived = self.compute_derived_metrics(runs_raw)
        for metric, values in derived.items():
            stats[metric]["min"] = values
            stats[metric]["max"] = values
            stats[metric]["avg"] = values

        # Build results table
        rows = []
        for metric, stat in sorted(stats.items()):
            min_vals = [v for v in stat["min"] if v != DEFAULT_VALUE]
            max_vals = [v for v in stat["max"] if v != DEFAULT_VALUE]
            avg_vals = [v for v in stat["avg"] if v != DEFAULT_VALUE]
            unit = METRIC_UNITS.get(metric, "")
            name = f"{metric}_{unit}" if unit else metric
            row = {
                "metric": name,
                "avg_min": (
                    round(float(np.mean(min_vals)), self.decimal_places)
                    if min_vals
                    else DEFAULT_VALUE
                ),
                "std_min": (
                    round(float(np.std(min_vals, ddof=0)), self.decimal_places)
                    if min_vals
                    else DEFAULT_VALUE
                ),
                "avg_avg": (
                    round(float(np.mean(avg_vals)), self.decimal_places)
                    if avg_vals
                    else DEFAULT_VALUE
                ),
                "std_avg": (
                    round(float(np.std(avg_vals, ddof=0)), self.decimal_places)
                    if avg_vals
                    else DEFAULT_VALUE
                ),
                "avg_max": (
                    round(float(np.mean(max_vals)), self.decimal_places)
                    if max_vals
                    else DEFAULT_VALUE
                ),
                "std_max": (
                    round(float(np.std(max_vals, ddof=0)), self.decimal_places)
                    if max_vals
                    else DEFAULT_VALUE
                ),
            }
            rows.append(row)

        return pl.DataFrame(rows)

    def save(self) -> Path:
        """Run analysis and save results.csv to experiment directory."""
        df = self.analyze()
        if df.is_empty():
            logger.warning("No results to save for %s", self.experiment_dir)
            return self.experiment_dir / "results.csv"

        output_path = self.experiment_dir / "results.csv"
        df.write_csv(output_path)
        logger.info("Results saved to %s", output_path)
        return output_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Per-experiment analysis: aggregate across runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        required=True,
        help="Path to experiment directory (e.g., runs/exp19)",
    )
    parser.add_argument(
        "--agg-mode",
        "-a",
        type=str,
        default="min",
        choices=("min", "max", "mean", "last", "median"),
        help="Per-run aggregation mode",
    )
    parser.add_argument(
        "--decimal-places",
        "-d",
        type=int,
        default=4,
        help="Number of decimal places",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    analysis = ExperimentAnalysis(
        experiment_dir=args.experiment,
        agg_mode=args.agg_mode,
        decimal_places=args.decimal_places,
    )
    df = analysis.analyze()
    if not df.is_empty():
        output_path = analysis.save()
        sys.stdout.buffer.write(
            (f"Saved to {output_path}\n" + str(df) + "\n").encode(
                "utf-8", errors="replace"
            )
        )
    else:
        logger.warning("No results generated")


if __name__ == "__main__":
    main()
