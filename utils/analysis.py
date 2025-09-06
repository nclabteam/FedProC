import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import polars as pl


def _load_experiment_data(experiment_dir):
    """
    Load experiment data from a single experiment directory.

    Args:
        experiment_dir (str): Path to experiment directory

    Returns:
        dict: Dictionary containing config, client info, and runs data
    """
    datum = {}

    # Load config.json
    config_path = os.path.join(experiment_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            datum.update(config)

    # Load info.json
    info_path = os.path.join(experiment_dir, "info.json")
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            info = json.load(f)
            datum["clients"] = info

    # Load runs
    runs = []
    for run_file in os.listdir(experiment_dir):
        p = os.path.join(experiment_dir, run_file)
        if not os.path.isdir(p):
            continue
        pr = os.path.join(p, "results", "server.csv")
        if not os.path.exists(pr):
            continue
        f = pl.read_csv(pr).to_dict(as_series=False)
        r = {"name": run_file, **f}
        runs.append(r)

    # Calculate average of all runs
    if runs:
        avg_run = _calculate_average_run(runs)
        runs.append(avg_run)

    datum["runs"] = runs
    return datum


def _calculate_average_run(runs):
    """
    Calculate the average values across all runs.

    Args:
        runs (list): List of run dictionaries containing metrics

    Returns:
        dict: Dictionary with averaged values and name "avg"
    """
    if not runs:
        return {"name": "avg"}

    # Get all metric keys (excluding 'name')
    metric_keys = set()
    for run in runs:
        metric_keys.update(key for key in run.keys() if key != "name")

    avg_run = {"name": "avg"}

    for metric in metric_keys:
        # Collect values for this metric from all runs
        values = []
        for run in runs:
            if metric in run and run[metric] is not None:
                metric_values = run[metric]
                # Handle both single values and lists
                if isinstance(metric_values, list):
                    # If it's a list, we need to average element-wise
                    values.append(metric_values)
                else:
                    # Single value
                    values.append([metric_values])

        if values:
            # Convert to numpy arrays for easier averaging
            if all(isinstance(v, list) for v in values):
                # Check if all lists have the same length
                lengths = [len(v) for v in values]
                if len(set(lengths)) == 1:  # All same length
                    # Element-wise average
                    avg_values = np.mean(values, axis=0).tolist()
                    # If it was originally a single value, keep it as single value
                    if len(avg_values) == 1:
                        avg_run[metric] = avg_values[0]
                    else:
                        avg_run[metric] = avg_values
                else:
                    # Different lengths, skip this metric or handle differently
                    print(
                        f"Warning: Inconsistent lengths for metric '{metric}', skipping average calculation"
                    )
                    avg_run[metric] = None
            else:
                # Simple average for non-list values
                numeric_values = []
                for v in values:
                    try:
                        if isinstance(v, list) and len(v) == 1:
                            numeric_values.append(float(v[0]))
                        else:
                            numeric_values.append(float(v))
                    except (ValueError, TypeError):
                        continue

                if numeric_values:
                    avg_run[metric] = np.mean(numeric_values)
                else:
                    avg_run[metric] = None

    return avg_run


def load_all_experiments(runs_dir="runs"):
    """
    Load all valid experiments from the runs directory.

    Args:
        runs_dir (str): Directory containing experiment folders

    Returns:
        list: List of experiment data dictionaries
    """
    experiments = []

    if not os.path.exists(runs_dir):
        print(f"Runs directory '{runs_dir}' not found")
        return experiments

    for path in os.listdir(runs_dir):
        p = os.path.join(runs_dir, path)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "results.csv")):
            datum = _load_experiment_data(p)
            datum["experiment_name"] = path
            experiments.append(datum)

    return experiments


def get_experiment_paths(experiments):
    """
    Extract experiment paths from experiment data.

    Args:
        experiments (list): List of experiment dictionaries

    Returns:
        list: List of experiment names/paths
    """
    return [exp["experiment_name"] for exp in experiments]


def extract_loss_values(run, loss_metric, loss_aggregation="min"):
    """
    Extract loss values from a run dictionary using specified aggregation.

    Args:
        run (dict): Run data dictionary
        loss_metric (str): Name of the loss metric to extract
        loss_aggregation (str): How to aggregate loss values - "min", "max", "mean", "last", or "median"

    Returns:
        float or None: Aggregated loss value or None if not found
    """
    if loss_metric not in run:
        return None

    loss_data = run[loss_metric]
    if isinstance(loss_data, list) and len(loss_data) > 0:
        if loss_aggregation == "min":
            # Take the minimum value from the entire training
            return min(loss_data)
        elif loss_aggregation == "max":
            # Take the maximum value from the entire training
            return max(loss_data)
        elif loss_aggregation == "mean":
            # Take the average value across all epochs
            return sum(loss_data) / len(loss_data)
        elif loss_aggregation == "last":
            # Take the last value (final convergence)
            return loss_data[-1]
        elif loss_aggregation == "median":
            # Take the median value across all epochs
            sorted_data = sorted(loss_data)
            n = len(sorted_data)
            if n % 2 == 0:
                return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
            else:
                return sorted_data[n // 2]
        else:
            # Default to minimum
            return min(loss_data)
    elif isinstance(loss_data, (int, float)):
        return loss_data

    return None


def get_loss_metric_name(save_local_model):
    """
    Determine the appropriate loss metric name based on configuration.

    Args:
        save_local_model (bool): Whether local models are saved

    Returns:
        str: Loss metric name to use
    """
    return "personal_avg_test_loss" if save_local_model else "global_avg_test_loss"


def get_experiment_config(experiment):
    """
    Extract configuration parameters from an experiment.

    Args:
        experiment (dict): Experiment data dictionary

    Returns:
        dict: Configuration parameters
    """
    return {
        "strategy": experiment.get("strategy", "unknown"),
        "save_local_model": experiment.get("save_local_model", False),
        "dataset": experiment.get("dataset", "unknown"),
        "input_len": experiment.get(
            "input_length", experiment.get("input_len", "unknown")
        ),
        "output_len": experiment.get(
            "output_length", experiment.get("output_len", "unknown")
        ),
        "model": experiment.get("model", "unknown"),
    }


def get_experiment_names_from_excel(excel_path):
    df = pl.read_excel(excel_path)
    if "--name=" not in df.columns:
        raise ValueError("Excel file must have a '--name=' column.")
    return set(df["--name="].to_list())


def filter_experiments(
    experiments, models=None, strategies=None, datasets=None, experiments_filter=None
):
    filtered = experiments
    if models:
        filtered = [exp for exp in filtered if exp.get("model", "unknown") in models]
    if strategies:
        filtered = [
            exp for exp in filtered if exp.get("strategy", "unknown") in strategies
        ]
    if datasets:
        filtered = [
            exp for exp in filtered if exp.get("dataset", "unknown") in datasets
        ]
    if experiments_filter:
        filtered = [
            exp
            for exp in filtered
            if exp.get("experiment_name", "") in experiments_filter
        ]
    return filtered


def group_experiments_by_model(all_experiments):
    model_groups = defaultdict(list)
    for exp in all_experiments:
        model_name = exp.get("model", "unknown")
        model_groups[model_name].append(exp)
    return model_groups


def make_metadata_table(metadata_data):
    return pl.DataFrame(metadata_data) if metadata_data else None


def pivot_table(df, value_col, index_cols, on_col):
    return df.pivot(values=value_col, index=index_cols, on=on_col)


def create_ranking_table_from_pivot(
    main_df, tiebreak_df=None, decimal_places=3, sort_cols=None, std_multiplier=1.0
):
    """
    Create a ranking table from a main DataFrame and an optional tiebreak DataFrame.
    If tiebreak_df is None, use main_df for both mean and std (for time tables).
    Both mean and std are rounded before ranking.
    std_multiplier is applied to std before rounding and sorting.
    """
    strategy_columns = [
        col for col in main_df.columns if col not in ["model", "dataset", "in", "out"]
    ]
    if not strategy_columns:
        return None
    ranking_rows = []
    best_strategy_counts = Counter()
    for i in range(len(main_df)):
        row_data = {
            "dataset": main_df["dataset"][i],
            "in": main_df["in"][i],
            "out": main_df["out"][i],
        }
        strategy_scores = []
        for strategy in strategy_columns:
            main_val = main_df[strategy][i]
            if tiebreak_df is not None and strategy in tiebreak_df.columns:
                tiebreak_val = tiebreak_df[strategy][i]
            else:
                tiebreak_val = main_val  # Use main_val as std if tiebreak_df is None
            if main_val is not None and not np.isnan(main_val):
                rounded_main = round(main_val, decimal_places)
                # Multiply std by std_multiplier, then round
                rounded_tiebreak = round(
                    (tiebreak_val if tiebreak_val is not None else 0) * std_multiplier,
                    decimal_places,
                )
                strategy_scores.append((strategy, rounded_main, rounded_tiebreak))
        if not strategy_scores:
            continue
        strategy_scores.sort(key=lambda x: (x[1], x[2]))
        rankings = {}
        for rank, (strategy, _, _) in enumerate(strategy_scores, 1):
            rankings[strategy] = rank
        for strategy in strategy_columns:
            row_data[strategy] = rankings.get(strategy, "N/A")
        best_strategies = [s for s, r in rankings.items() if r == 1]
        if best_strategies:
            best_strategy = best_strategies[0]
            best_strategy_counts[best_strategy] += 1
            row_data["best_strategy"] = best_strategy
        else:
            row_data["best_strategy"] = "N/A"
        ranking_rows.append(row_data)
    # Average ranks and most frequent winner
    if ranking_rows:
        avg_ranks = {"dataset": "AVG_RANK", "in": "", "out": ""}
        strategy_avg_ranks = {}
        for strategy in strategy_columns:
            valid_ranks = [
                row[strategy]
                for row in ranking_rows
                if row[strategy] != "N/A" and isinstance(row[strategy], (int, float))
            ]
            if valid_ranks:
                avg_rank = round(np.mean(valid_ranks), 2)
                avg_ranks[strategy] = avg_rank
                strategy_avg_ranks[strategy] = avg_rank
            else:
                avg_ranks[strategy] = "N/A"
                strategy_avg_ranks[strategy] = float("inf")
        if best_strategy_counts:
            max_count = max(best_strategy_counts.values())
            top_strategies = [
                strategy
                for strategy, count in best_strategy_counts.items()
                if count == max_count
            ]
            if len(top_strategies) == 1:
                most_frequent = top_strategies[0]
                avg_ranks["best_strategy"] = f"{most_frequent} ({max_count}x)"
            else:
                best_by_avg_rank = min(
                    top_strategies,
                    key=lambda s: strategy_avg_ranks.get(s, float("inf")),
                )
                avg_ranks["best_strategy"] = f"{best_by_avg_rank} ({max_count}x)(tie)"
        else:
            avg_ranks["best_strategy"] = "N/A"
        ranking_rows.append(avg_ranks)
    if ranking_rows:
        ranking_df = pl.DataFrame(ranking_rows)
        if sort_cols:
            data_rows = ranking_df.filter(pl.col("dataset") != "AVG_RANK")
            avg_row = ranking_df.filter(pl.col("dataset") == "AVG_RANK")
            data_rows = data_rows.sort(sort_cols)
            ranking_df = pl.concat([data_rows, avg_row])
        return ranking_df
    return None


def parse_args(default_table_type="model-specific"):
    parser = argparse.ArgumentParser(
        description="Generate analysis tables from federated learning experiment results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--runs-dir",
        "-r",
        type=str,
        default="runs",
        help="Directory containing experiment folders",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="analysis/tables",
        help="Output directory for generated tables",
    )
    parser.add_argument(
        "--table-type",
        "-t",
        choices=["model-specific", "comparison", "both"],
        default=default_table_type,
        help="Type of tables to generate",
    )
    parser.add_argument(
        "--std-multiplier",
        "-s",
        type=float,
        default=10e3,
        help="Factor to multiply standard deviation for better visibility",
    )
    parser.add_argument(
        "--decimal-places",
        "-d",
        type=int,
        default=3,
        help="Number of decimal places to display in the results",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display tables to console, only save to files",
    )
    parser.add_argument(
        "--show-metadata",
        action="store_true",
        help="Display metadata table to console (metadata always saved to file)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Reduce output verbosity"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Filter to specific models (e.g., --models linear lstm)",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        help="Filter to specific strategies (e.g., --strategies fedavg fedprox)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        help="Filter to specific datasets (e.g., --datasets stock crypto)",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        help="Process only specific experiments (e.g., --experiments exp76 exp77)",
    )
    parser.add_argument(
        "--excel",
        type=str,
        help="Excel file to filter experiments (column '--name=' should contain experiment names)",
    )
    return parser.parse_args()


def filter_experiments(experiments, args):
    filtered = experiments
    if args.models:
        filtered = [
            exp for exp in filtered if exp.get("model", "unknown") in args.models
        ]
    if args.strategies:
        filtered = [
            exp for exp in filtered if exp.get("strategy", "unknown") in args.strategies
        ]
    if args.datasets:
        filtered = [
            exp for exp in filtered if exp.get("dataset", "unknown") in args.datasets
        ]
    if args.experiments:
        filtered = [
            exp
            for exp in filtered
            if exp.get("experiment_name", "") in args.experiments
        ]
    return filtered


if __name__ == "__main__":
    # Simple utility test - load and display basic experiment info
    experiments = load_all_experiments()

    print("Scanning experiments...")
    for exp in experiments:
        print(f"Experiment: {exp['experiment_name']}")
        print(f"Model: {exp.get('model', 'unknown')}")
        print(f"Strategy: {exp.get('strategy', 'unknown')}")
        print(f"Dataset: {exp.get('dataset', 'unknown')}")
        print(f"Number of runs: {len(exp.get('runs', []))}")

        # Print run names
        for run in exp.get("runs", []):
            print(f"  Run: {run['name']}")

        print("-" * 50)

    print(f"\nTotal experiments found: {len(experiments)}")
