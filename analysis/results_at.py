import json
import os

import numpy as np
import pandas as pd

# --- Configuration ---
# The performance metric to analyze.
TARGET_COLUMN = "global_avg_test_loss"
# The specific rounds (iterations) to check performance at.
MILESTONE_ROUNDS = [100, 200, 300, 400, 500]
# The number of decimal places for rounding the final output.
ROUND_PRECISION = 4
# The directory containing all experimental runs.
RUNS_DIRECTORY = "runs"
# The output file for this analysis.
OUTPUT_FILE = os.path.join("analysis", "experiment_summary.csv")


def analyze_experiment_with_pandas(experiment_dir: str):
    """
    Analyzes all trials in an experiment to produce a summary of performance
    at key milestones using Pandas.

    Args:
        experiment_dir (str): The name of the experiment directory.

    Returns:
        A Pandas DataFrame with the summarized results for the experiment, or None.
    """
    print(f"--- Analyzing Experiment: {experiment_dir} ---")
    base_path = os.path.join(RUNS_DIRECTORY, experiment_dir)

    # 1. Load experiment metadata from config.json
    config_path = os.path.join(base_path, "config.json")
    metadata = {}
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        metadata = {
            "dataset": config.get("dataset"),
            "model": config.get("model"),
            "input_len": config.get("input_len"),
            "output_len": config.get("output_len"),
        }
        print(f"  - Loaded metadata: {metadata}")
    except Exception as e:
        print(f"  - Warning: Could not read config.json in {base_path}. Reason: {e}")

    # 2. Load all trial CSVs into a list of DataFrames
    trial_dfs = []
    for i, trial_name in enumerate(os.listdir(base_path)):
        trial_path = os.path.join(base_path, trial_name)
        csv_path = os.path.join(trial_path, "results", "server.csv")
        if os.path.isdir(trial_path) and os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if TARGET_COLUMN in df.columns:
                    df["trial_id"] = i
                    df["round"] = df.index + 1  # Add a round number column
                    trial_dfs.append(df)
            except Exception as e:
                print(f"  - Error reading {csv_path}: {e}. Skipping trial.")

    if not trial_dfs:
        print(f"  - No valid trial data found. Skipping experiment.\n")
        return None

    print(f"  - Found {len(trial_dfs)} trials to analyze.")

    # 3. Combine all trials into a single master DataFrame for this experiment
    all_trials_df = pd.concat(trial_dfs, ignore_index=True)

    # 4. Group and aggregate data for the fixed milestone rounds
    milestones_df = all_trials_df[all_trials_df["round"].isin(MILESTONE_ROUNDS)]
    milestone_summary = (
        milestones_df.groupby("round")[TARGET_COLUMN]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # 5. Get the data for the 'last' round of each trial
    # We find the index of the last round for each trial_id and select those rows
    last_round_indices = all_trials_df.groupby("trial_id")["round"].idxmax()
    last_rounds_df = all_trials_df.loc[last_round_indices]

    # Calculate stats for the 'last' round
    last_round_summary = pd.DataFrame(
        [
            {
                "round": "last",
                "mean": last_rounds_df[TARGET_COLUMN].mean(),
                "std": last_rounds_df[TARGET_COLUMN].std(),
                "count": last_rounds_df[TARGET_COLUMN].count(),
            }
        ]
    )

    # 6. Combine the milestone and last-round summaries
    full_summary = pd.concat([milestone_summary, last_round_summary], ignore_index=True)

    # Rename columns for clarity
    full_summary = full_summary.rename(
        columns={
            "round": "milestone_round",
            "mean": "mean_loss",
            "std": "std_loss",
            "count": "num_trials_at_milestone",
        }
    )

    # 7. Add metadata and case name to the summary DataFrame
    full_summary["case"] = experiment_dir
    for key, value in metadata.items():
        full_summary[key] = value

    # Reorder columns for a clean final output
    final_columns = [
        "case",
        "dataset",
        "model",
        "input_len",
        "output_len",
        "milestone_round",
        "mean_loss",
        "std_loss",
        "num_trials_at_milestone",
    ]

    print("  - Analysis complete.\n")
    return full_summary[final_columns]


def main():
    """Main function to run the entire analysis pipeline."""
    output_dir = os.path.dirname(OUTPUT_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.isdir(RUNS_DIRECTORY):
        print(f"Error: The '{RUNS_DIRECTORY}' directory does not exist.")
        return

    all_experiment_results = []
    for experiment_dir in sorted(os.listdir(RUNS_DIRECTORY)):
        if os.path.isdir(os.path.join(RUNS_DIRECTORY, experiment_dir)):
            result_df = analyze_experiment_with_pandas(experiment_dir)
            if result_df is not None and not result_df.empty:
                all_experiment_results.append(result_df)

    if not all_experiment_results:
        print(
            "No experiments were successfully analyzed. No output file will be created."
        )
        return

    # Create the final master DataFrame and save it
    final_df = pd.concat(all_experiment_results, ignore_index=True)
    final_df = final_df.round(ROUND_PRECISION)

    try:
        final_df.to_csv(OUTPUT_FILE, index=False)
        print("=" * 60)
        print(f"✅ Experiment summary successfully saved to: {OUTPUT_FILE}")
        print("=" * 60)
        print("Final DataFrame preview:")
        print(final_df.to_string())
    except Exception as e:
        print(f"Error: Could not write final CSV to {OUTPUT_FILE}. Reason: {e}")


if __name__ == "__main__":
    main()
