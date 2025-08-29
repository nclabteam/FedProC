import json
import os
import sys

import numpy as np
import polars as pl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


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
