import json
import os

import numpy as np
import polars as pl

# --- Configuration ---
# The column to analyze.
TARGET_COLUMN = "global_avg_test_loss"
# The specific rounds (iterations) to check performance at.
MILESTONE_ROUNDS = [100, 200, 300, 400, 500]
# The number of decimal places to round floats to.
ROUND_PRECISION = 4
# The directory containing all experimental runs.
RUNS_DIRECTORY = "runs"
# The output file for this new analysis.
OUTPUT_FILE = os.path.join("analysis", "loss_at_milestones.csv")


def perform_milestone_analysis(experiment_dir: str):
    """
    Analyzes the mean and std of loss at specific milestone rounds for an experiment.
    """
    print(f"--- Analyzing Milestones for: {experiment_dir} ---")
    base_path = os.path.join(RUNS_DIRECTORY, experiment_dir)

    # Load experiment metadata
    config_path = os.path.join(base_path, "config.json")
    metadata = {}
    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)
        metadata = {
            "dataset": config_data.get("dataset"),
            "model": config_data.get("model"),
            "input_len": config_data.get("input_len"),
            "output_len": config_data.get("output_len"),
        }
        print(f"  - Loaded metadata: {metadata}")
    except Exception as e:
        print(f"  - Warning: Could not read config.json in {base_path}. Reason: {e}")

    # 1. Load the loss data from each trial
    trial_losses = []
    for trial_name in os.listdir(base_path):
        trial_path = os.path.join(base_path, trial_name)
        csv_path = os.path.join(trial_path, "results", "server.csv")
        if os.path.isdir(trial_path) and os.path.exists(csv_path):
            try:
                df = pl.read_csv(csv_path)
                if TARGET_COLUMN in df.columns:
                    trial_losses.append(df.get_column(TARGET_COLUMN))
            except Exception as e:
                print(f"  - Error reading {csv_path}: {e}. Skipping trial.")

    if not trial_losses:
        print(
            f"  - No valid trial data found for experiment '{experiment_dir}'. Skipping.\n"
        )
        return None

    print(f"  - Found {len(trial_losses)} trials to analyze.")
    milestone_results = []

    # 2. Analyze performance at fixed milestone rounds (100, 200, etc.)
    for milestone_round in MILESTONE_ROUNDS:
        losses_at_milestone = []
        for loss_series in trial_losses:
            # Check if the trial ran for at least this many rounds
            if len(loss_series) >= milestone_round:
                # Get loss at the specific round (index is round - 1)
                losses_at_milestone.append(loss_series[milestone_round - 1])

        # Calculate mean and std if any trials reached this milestone
        if losses_at_milestone:
            mean_loss = np.mean(losses_at_milestone)
            std_loss = np.std(losses_at_milestone)
            num_trials_at_milestone = len(losses_at_milestone)
        else:
            mean_loss, std_loss, num_trials_at_milestone = None, None, 0

        milestone_results.append(
            {
                "case": experiment_dir,
                **metadata,
                "milestone_round": milestone_round,
                "mean_loss": mean_loss,
                "std_loss": std_loss,
                "num_trials_at_milestone": num_trials_at_milestone,
                "total_trials": len(trial_losses),
            }
        )

    # 3. Analyze performance at the LAST round for every trial
    last_round_losses = [s[-1] for s in trial_losses if len(s) > 0]
    if last_round_losses:
        mean_loss = np.mean(last_round_losses)
        std_loss = np.std(last_round_losses)
        num_trials_at_milestone = len(last_round_losses)

        milestone_results.append(
            {
                "case": experiment_dir,
                **metadata,
                "milestone_round": "last",
                "mean_loss": mean_loss,
                "std_loss": std_loss,
                "num_trials_at_milestone": num_trials_at_milestone,
                "total_trials": len(trial_losses),
            }
        )

    print(f"  - Milestone analysis complete.\n")
    return pl.DataFrame(milestone_results)


def main():
    """Main function to run the entire milestone analysis pipeline."""
    output_dir = os.path.dirname(OUTPUT_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.isdir(RUNS_DIRECTORY):
        print(f"Error: The '{RUNS_DIRECTORY}' directory does not exist.")
        return

    all_experiments_df = []
    for experiment_dir in sorted(os.listdir(RUNS_DIRECTORY)):
        if os.path.isdir(os.path.join(RUNS_DIRECTORY, experiment_dir)):
            experiment_df = perform_milestone_analysis(experiment_dir)
            if experiment_df is not None and not experiment_df.is_empty():
                all_experiments_df.append(experiment_df)

    if not all_experiments_df:
        print(
            "No experiments were successfully analyzed. No output file will be created."
        )
        return

    final_df = pl.concat(all_experiments_df)

    float_columns = final_df.select(pl.col(pl.Float64)).columns
    final_df = final_df.with_columns(pl.col(float_columns).round(ROUND_PRECISION))

    try:
        final_df.write_csv(OUTPUT_FILE)
        print("=" * 50)
        print(f"✅ Milestone analysis successfully saved to: {OUTPUT_FILE}")
        print("=" * 50)
        print("Final DataFrame preview:")
        print(final_df)
    except Exception as e:
        print(f"Error: Could not write final CSV to {OUTPUT_FILE}. Reason: {e}")


if __name__ == "__main__":
    main()
