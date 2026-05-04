"""
Cross-experiment analysis.

Reads results.csv from multiple experiments, builds pivot and ranking tables.

Input:  runs/ directory (multiple experiments)
Output: analysis/tables/ (pivot CSVs, ranking CSVs)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import polars as pl

sys.path.append(str(Path(__file__).resolve().parent.parent))

from analysis.io import (
    SIZE_UNITS,
    TIME_UNITS,
    convert_size,
    convert_time,
    load_config,
    read_csv,
    resolve_loss_metric,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

pl.Config.set_tbl_cols(100)
pl.Config.set_tbl_rows(100)

# Type aliases
AGG_MODES = ("min", "max", "mean", "last", "median")
AggMode = Literal["min", "max", "mean", "last", "median"]
PIVOT_OPTIONS = ("model", "strategy")
PivotOption = Literal["model", "strategy"]
MISSING_VALUE: float = 0.0

# Identifier fields (not hyperparameters)
IDENTIFIER_FIELDS = frozenset(
    {
        "exp",
        "aggregates",
        "dataset",
        "input_len",
        "output_len",
        "model",
        "strategy",
        "name",
    }
)

# Available metrics
METRICS = (
    "all",
    "loss",
    "personalization_loss",
    "generalization_loss",
    "efficiency",
    "communication",
    "time_per_experiment",
    "last_improvement_round",
    "longest_improvement_streak",
    "most_frequent_improvement_streak",
    "oscillation_count",
    "improvement_ratio",
    "improvement_magnitude",
    "hyperparameters",
)

METRIC_DESCRIPTIONS = {
    "loss": {"title": "Test loss", "explanation": "Lower is better."},
    "personalization_loss": {
        "title": "Personalization test loss",
        "explanation": "personal_avg_test_loss.",
    },
    "generalization_loss": {
        "title": "Generalization test loss",
        "explanation": "global_avg_test_loss.",
    },
    "efficiency": {
        "title": "Time per round",
        "explanation": "Lower = faster training.",
    },
    "communication": {
        "title": "Total communication cost",
        "explanation": "Lower = less bandwidth.",
    },
    "time_per_experiment": {
        "title": "Wall-clock time per run",
        "explanation": "Total time for one full run.",
    },
    "last_improvement_round": {
        "title": "Last improvement round",
        "explanation": "When the model stopped improving.",
    },
    "longest_improvement_streak": {
        "title": "Longest improvement streak",
        "explanation": "Max consecutive rounds of improvement.",
    },
    "most_frequent_improvement_streak": {
        "title": "Most frequent streak",
        "explanation": "Typical improvement pattern.",
    },
    "oscillation_count": {
        "title": "Oscillation count",
        "explanation": "Direction changes in loss. Lower = more stable.",
    },
    "improvement_ratio": {
        "title": "Improvement ratio",
        "explanation": "Fraction of rounds with improvement (0-1).",
    },
    "improvement_magnitude": {
        "title": "Improvement magnitude",
        "explanation": "Avg loss reduction per improvement.",
    },
}


class ExperimentComparison:
    """Compare results across multiple experiments."""

    def __init__(
        self,
        runs_dir: str | Path,
        output_dir: str | Path = "analysis/tables",
        std_multiplier: float = 1.0,
        decimal_places: int = 3,
        agg_mode: AggMode = "min",
        time_unit: str = "s",
        size_unit: str = "mb",
    ) -> None:
        self.runs_dir = Path(runs_dir)
        self.output_dir = Path(output_dir)
        self.std_multiplier = std_multiplier
        self.decimal_places = decimal_places
        self.agg_mode = agg_mode
        self.time_unit = time_unit.lower()
        self.size_unit = size_unit.lower()

    # =========================================================================
    # Helpers
    # =========================================================================

    def _format_mean(self, value: float) -> float:
        return round(value, self.decimal_places)

    def _format_std(self, value: float) -> float:
        return round(value * self.std_multiplier, self.decimal_places)

    def _format_mean_std_str(self, mean: float, std: float) -> str:
        return f"{mean}+/-{std}"

    def _convert_time(self, value: float) -> float:
        return convert_time(value, from_unit="s", to_unit=self.time_unit)

    def _convert_size(self, value: float) -> float:
        return convert_size(value, from_unit="mb", to_unit=self.size_unit)

    def _resolve_metric(self, metric: str, experiment: Dict) -> str:
        if metric == "personalization_loss":
            return "personal_avg_test_loss"
        if metric == "generalization_loss":
            return "global_avg_test_loss"
        if metric == "loss":
            return resolve_loss_metric(experiment.get("save_local_model", False))
        return metric

    def _get_metric_value_str(self, experiment: Dict, metric: str) -> Optional[str]:
        resolved = self._resolve_metric(metric, experiment)
        aggs = experiment.get("aggregates", {})
        mean_val = aggs.get(f"{resolved}_{self.agg_mode}_mean")
        std_val = aggs.get(f"{resolved}_{self.agg_mode}_std")
        if mean_val is None or std_val is None:
            return None
        if mean_val == MISSING_VALUE and std_val == MISSING_VALUE:
            return None
        return self._format_mean_std_str(mean_val, std_val)

    def _get_metric_mean_std(
        self, experiment: Dict, metric: str
    ) -> Optional[Tuple[float, float]]:
        resolved = self._resolve_metric(metric, experiment)
        aggs = experiment.get("aggregates", {})
        mean_val = aggs.get(f"{resolved}_{self.agg_mode}_mean")
        std_val = aggs.get(f"{resolved}_{self.agg_mode}_std")
        if mean_val is None or std_val is None:
            return None
        if mean_val == MISSING_VALUE and std_val == MISSING_VALUE:
            return None
        return (mean_val, std_val)

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
        exact: bool = False,
    ) -> bool:
        def _check_match(target: str, queries: List[str]) -> bool:
            target_lower = target.lower()
            if exact:
                return any(q.lower() == target_lower for q in queries)
            return any(q.lower() in target_lower for q in queries)

        if experiments is not None and not _check_match(
            experiment.get("exp", ""), experiments
        ):
            return False
        if models is not None and not _check_match(experiment.get("model", ""), models):
            return False
        if strategies is not None and not _check_match(
            experiment.get("strategy", ""), strategies
        ):
            return False
        if datasets is not None and not _check_match(
            experiment.get("dataset", ""), datasets
        ):
            return False
        if output_lens is not None:
            exp_output_len = experiment.get("output_len")
            if exp_output_len is None or int(exp_output_len) not in output_lens:
                return False
        return True

    # =========================================================================
    # Loading
    # =========================================================================

    def load_experiments(
        self,
        models: Optional[List[str]] = None,
        strategies: Optional[List[str]] = None,
        datasets: Optional[List[str]] = None,
        output_lens: Optional[List[int]] = None,
        experiments: Optional[List[str]] = None,
        exact: bool = False,
    ) -> List[Dict]:
        """Load results.csv + config.json from each experiment."""
        all_experiments = []

        if not self.runs_dir.exists():
            logger.warning("Runs directory '%s' not found", self.runs_dir)
            return all_experiments

        for child in sorted(self.runs_dir.iterdir()):
            if not child.is_dir():
                continue
            if not (child / "config.json").exists():
                continue
            if not (child / "results.csv").exists():
                logger.debug("Skipping (no results.csv): %s", child)
                continue

            datum = {"exp": child.name}
            datum.update(load_config(child))

            if not self._matches_filter(
                experiment=datum,
                models=models,
                strategies=strategies,
                datasets=datasets,
                output_lens=output_lens,
                experiments=experiments,
                exact=exact,
            ):
                continue

            # Load results.csv (produced by single.py)
            results_df = read_csv(child / "results.csv")
            if results_df is None:
                continue

            # Build aggregates dict from results.csv
            aggregates = {}
            metrics_list = results_df.get("metric", [])
            for i, metric_name in enumerate(metrics_list):
                # Strip unit suffix (e.g. "efficiency_s" -> "efficiency")
                base_name = metric_name
                for suffix in ("_s", "_MB", "_GB", "_round", "_count", "_ratio"):
                    if metric_name.endswith(suffix):
                        base_name = metric_name[: -len(suffix)]
                        break
                for stat_col in ("avg_min", "std_min", "avg_max", "std_max"):
                    val = results_df.get(stat_col, [None])[i]
                    if val is not None:
                        # Map to the key format expected: {metric}_{agg_mode}_{mean|std}
                        if stat_col.startswith("avg"):
                            aggregates[f"{base_name}_{self.agg_mode}_mean"] = val
                        else:
                            aggregates[f"{base_name}_{self.agg_mode}_std"] = val

            datum["aggregates"] = aggregates
            all_experiments.append(datum)

        return all_experiments

    # =========================================================================
    # Pivot table
    # =========================================================================

    def _get_pivot_config(self, pivot: PivotOption) -> Tuple[str, str, Tuple[str, ...]]:
        if pivot == "model":
            return "model", "strategy", ("dataset", "input_len", "output_len")
        return "strategy", "model", ("dataset", "input_len", "output_len")

    def create_pivot_table(
        self,
        experiments: List[Dict],
        metric: str = "loss",
        group_by: Tuple[str, ...] = ("dataset", "input_len", "output_len"),
        pivot_by: str = "strategy",
    ) -> pl.DataFrame:
        """Create a pivot table with mean±std values."""
        rows = []
        for exp in experiments:
            row = {field: exp.get(field) for field in group_by}
            row[pivot_by] = exp.get(pivot_by)
            row["value"] = self._get_metric_value_str(exp, metric)
            rows.append(row)

        if not rows:
            return pl.DataFrame()

        df = pl.DataFrame(rows)
        pivot_values = df.select(pivot_by).unique().sort(pivot_by).to_series().to_list()

        df_pivot = df.pivot(
            on=pivot_by,
            index=list(group_by),
            values="value",
            aggregate_function="first",
        )
        col_order = list(group_by) + [c for c in pivot_values if c in df_pivot.columns]
        return df_pivot.select(col_order).sort(list(group_by))

    # =========================================================================
    # Ranking table
    # =========================================================================

    def create_ranking_table(
        self,
        experiments: List[Dict],
        metric: str = "loss",
        group_by: Tuple[str, ...] = ("dataset", "input_len", "output_len"),
        pivot_by: str = "strategy",
        lower_is_better: bool = True,
    ) -> pl.DataFrame:
        """Create a ranking table (1=best)."""
        config_data: Dict[Tuple, Dict[str, Tuple[float, float]]] = {}
        all_pivot_values: set = set()

        for exp in experiments:
            config_key = tuple(exp.get(f) for f in group_by)
            pv = exp.get(pivot_by)
            if pv is None:
                continue
            all_pivot_values.add(pv)
            mean_std = self._get_metric_mean_std(exp, metric)
            if mean_std is None:
                continue
            config_data.setdefault(config_key, {})[pv] = mean_std

        if not config_data:
            return pl.DataFrame()

        pivot_values = sorted(all_pivot_values)
        best_col = f"best_{pivot_by}"
        pivot_ranks: Dict[str, List[float]] = {pv: [] for pv in pivot_values}
        win_counts: Dict[str, int] = {pv: 0 for pv in pivot_values}
        rows = []

        for config_key in sorted(config_data):
            row = {field: config_key[i] for i, field in enumerate(group_by)}
            config_pivot_values = config_data[config_key]

            sorted_pv = sorted(
                config_pivot_values.items(),
                key=lambda x: (x[1][0], x[1][1]),
                reverse=not lower_is_better,
            )

            ranks: Dict[str, int] = {}
            current_rank = 1
            prev = None
            for i, (pv, mean_std) in enumerate(sorted_pv):
                if prev is not None and mean_std == prev:
                    ranks[pv] = ranks[sorted_pv[i - 1][0]]
                else:
                    ranks[pv] = current_rank
                prev = mean_std
                current_rank = i + 2

            best_pivot = None
            best_rank = float("inf")
            for pv in pivot_values:
                if pv in ranks:
                    row[pv] = str(ranks[pv])
                    pivot_ranks[pv].append(float(ranks[pv]))
                    if ranks[pv] < best_rank:
                        best_rank = ranks[pv]
                        best_pivot = pv
                else:
                    row[pv] = "N/A"

            if best_pivot:
                win_counts[best_pivot] += 1
                row[best_col] = best_pivot
            else:
                row[best_col] = "N/A"
            rows.append(row)

        # Average rank row
        avg_row = {group_by[0]: "AVG_RANK"}
        for f in group_by[1:]:
            avg_row[f] = ""
        for pv in pivot_values:
            avg_row[pv] = (
                f"{np.mean(pivot_ranks[pv]):.2f}" if pivot_ranks[pv] else "N/A"
            )

        # Most frequent winner
        max_wins = max(win_counts.values()) if win_counts else 0
        winners = [pv for pv, c in win_counts.items() if c == max_wins and c > 0]
        if len(winners) == 1:
            avg_row[best_col] = f"{winners[0]} ({max_wins}x)"
        elif len(winners) > 1:
            winner_avgs = [
                (w, np.mean(pivot_ranks[w]) if pivot_ranks[w] else float("inf"))
                for w in winners
            ]
            winner_avgs.sort(key=lambda x: x[1])
            best_avg = winner_avgs[0][1]
            final = [w for w, a in winner_avgs if a == best_avg]
            if len(final) == 1:
                avg_row[best_col] = f"{final[0]} ({max_wins}x)"
            else:
                avg_row[best_col] = (
                    f"tie: {', '.join(f'{w} ({max_wins}x)' for w in final)}"
                )
        else:
            avg_row[best_col] = "N/A"

        rows.append(avg_row)
        df = pl.DataFrame(rows)
        col_order = list(group_by) + pivot_values + [best_col]
        return df.select([c for c in col_order if c in df.columns])

    # =========================================================================
    # Display
    # =========================================================================

    def _get_table_header(
        self, group_value: str, metric: str, pivot: PivotOption
    ) -> str:
        group_label = "MODEL" if pivot == "model" else "STRATEGY"
        pivot_label = "strategy" if pivot == "model" else "model"
        lines = [
            "=" * 80,
            f"{group_label}: {group_value.upper()}",
            f"METRIC: {metric}",
        ]
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
                f"(std x{self.std_multiplier}, {self.decimal_places}dp, agg={self.agg_mode})",
                "=" * 80,
            ]
        )
        return "\n".join(lines)

    def _get_ranking_header(
        self, group_value: str, metric: str, pivot: PivotOption
    ) -> str:
        group_label = "MODEL" if pivot == "model" else "STRATEGY"
        pivot_label = "strategy" if pivot == "model" else "model"
        return "\n".join(
            [
                "=" * 80,
                f"RANKING: {group_label} {group_value.upper()} — {metric}",
                f"{pivot_label}s ranked by mean (1=best), ties broken by std",
                "=" * 80,
            ]
        )

    # =========================================================================
    # Markdown
    # =========================================================================

    def _escape_md(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    def _df_to_markdown(self, df: pl.DataFrame) -> str:
        if df.is_empty():
            return "_No data_"
        cols = df.columns
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = []
        for row in df.iter_rows(named=True):
            cells = [self._escape_md(row.get(c)) for c in cols]
            rows.append("| " + " | ".join(cells) + " |")
        return "\n".join([header, sep, *rows])

    # =========================================================================
    # Save
    # =========================================================================

    def save_tables(
        self,
        experiments: List[Dict],
        metric: str = "loss",
        pivot: PivotOption = "model",
        include_ranking: bool = True,
        lower_is_better: bool = True,
        markdown_path: Optional[str | Path] = None,
    ) -> None:
        """Save pivot and ranking tables grouped by pivot field."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        md_path = Path(markdown_path) if markdown_path else None
        if md_path:
            md_path.parent.mkdir(parents=True, exist_ok=True)

        if metric == "hyperparameters":
            return  # TODO: hyperparameter display

        group_by_field, pivot_by_field, row_group_by = self._get_pivot_config(pivot)
        group_values = sorted(
            set(exp.get(group_by_field, "unknown") for exp in experiments)
        )

        for gv in group_values:
            group_exps = [e for e in experiments if e.get(group_by_field) == gv]
            if not group_exps:
                continue

            # Pivot table
            df = self.create_pivot_table(
                group_exps, metric, row_group_by, pivot_by_field
            )
            if df.is_empty():
                continue

            header = self._get_table_header(gv, metric, pivot)
            sys.stdout.buffer.write((header + "\n").encode("utf-8", errors="replace"))
            sys.stdout.buffer.write(
                (str(df) + "\n\n").encode("utf-8", errors="replace")
            )

            filename = f"{gv}_{metric}_{self.agg_mode}.csv"
            df.write_csv(self.output_dir / filename)

            if md_path:
                with md_path.open("a", encoding="utf-8") as f:
                    f.write(f"## {gv} — {metric}\n\n")
                    f.write(self._df_to_markdown(df))
                    f.write("\n\n")

            # Ranking table
            if include_ranking:
                df_rank = self.create_ranking_table(
                    group_exps,
                    metric,
                    row_group_by,
                    pivot_by_field,
                    lower_is_better,
                )
                if not df_rank.is_empty():
                    rank_header = self._get_ranking_header(gv, metric, pivot)
                    sys.stdout.buffer.write(
                        (rank_header + "\n").encode("utf-8", errors="replace")
                    )
                    sys.stdout.buffer.write(
                        (str(df_rank) + "\n\n").encode("utf-8", errors="replace")
                    )

                    rank_filename = f"{gv}_{metric}_{self.agg_mode}_ranking.csv"
                    df_rank.write_csv(self.output_dir / rank_filename)

                    if md_path:
                        with md_path.open("a", encoding="utf-8") as f:
                            f.write(f"### Ranking — {gv}\n\n")
                            f.write(self._df_to_markdown(df_rank))
                            f.write("\n\n")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-experiment analysis: pivot and ranking tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--runs-dir", "-r", type=str, default="runs")
    parser.add_argument("--output-dir", "-o", type=str, default="analysis/tables")
    parser.add_argument("--std-multiplier", "-s", type=float, default=1e3)
    parser.add_argument("--decimal-places", "-d", type=int, default=3)
    parser.add_argument("--agg-mode", "-a", type=str, default="min", choices=AGG_MODES)
    parser.add_argument("--time-unit", "-t", type=str, default="s", choices=TIME_UNITS)
    parser.add_argument("--size-unit", "-z", type=str, default="mb", choices=SIZE_UNITS)
    parser.add_argument("--metric", "-m", type=str, default="loss", choices=METRICS)
    parser.add_argument(
        "--pivot", "-p", type=str, default="model", choices=PIVOT_OPTIONS
    )
    parser.add_argument("--no-ranking", action="store_true")
    parser.add_argument("--higher-is-better", action="store_true")
    parser.add_argument("--markdown-path", type=str, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")

    # Filters
    parser.add_argument("--models", type=str, nargs="+")
    parser.add_argument("--strategies", type=str, nargs="+")
    parser.add_argument("--datasets", type=str, nargs="+")
    parser.add_argument("--output-lens", type=int, nargs="+")
    parser.add_argument("--experiments", type=str, nargs="+")
    parser.add_argument("--exact", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    comparison = ExperimentComparison(
        runs_dir=args.runs_dir,
        output_dir=args.output_dir,
        std_multiplier=args.std_multiplier,
        decimal_places=args.decimal_places,
        agg_mode=args.agg_mode,
        time_unit=args.time_unit,
        size_unit=args.size_unit,
    )

    experiments = comparison.load_experiments(
        models=args.models,
        strategies=args.strategies,
        datasets=args.datasets,
        output_lens=args.output_lens,
        experiments=args.experiments,
        exact=args.exact,
    )

    if not experiments:
        logger.warning("No experiments loaded")
        return

    logger.info("Loaded %d experiments", len(experiments))

    if args.markdown_path:
        md = Path(args.markdown_path)
        md.parent.mkdir(parents=True, exist_ok=True)
        with md.open("w", encoding="utf-8") as f:
            f.write("# Analysis Tables\n\n")

    metrics_to_process = (
        [m for m in METRICS if m != "all"] if args.metric == "all" else [args.metric]
    )

    for metric in metrics_to_process:
        comparison.save_tables(
            experiments,
            metric=metric,
            pivot=args.pivot,
            include_ranking=not args.no_ranking,
            lower_is_better=not args.higher_is_better,
            markdown_path=args.markdown_path,
        )


if __name__ == "__main__":
    main()
