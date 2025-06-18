import os

import numpy as np
import polars as pl

# --- Configuration ---
# The column to analyze for convergence.
TARGET_COLUMN = "global_avg_test_loss"
# The step size for creating loss thresholds.
LOSS_STEP = 0.01
# The directory containing all experimental runs.
RUNS_DIRECTORY = "runs"
# The output file for the analysis.
OUTPUT_FILE = os.path.join("analysis", "convergence.csv")


def perform_convergence_analysis(experiment_dir: str):
    """
    Analyzes the convergence for all trials within a single experiment directory.

    Args:
        experiment_dir (str): The name of the experiment directory inside 'runs/'.

    Returns:
        A Polars DataFrame with the convergence analysis for the experiment,
        or None if no valid data is found.
    """
    print(f"--- Analyzing Experiment: {experiment_dir} ---")
    base_path = os.path.join(RUNS_DIRECTORY, experiment_dir)
    trial_losses = []

    # 1. Load the target loss column from each trial's server.csv
    for trial_name in os.listdir(base_path):
        trial_path = os.path.join(base_path, trial_name)
        csv_path = os.path.join(trial_path, "results", "server.csv")

        if os.path.isdir(trial_path) and os.path.exists(csv_path):
            try:
                df = pl.read_csv(csv_path)
                if TARGET_COLUMN in df.columns:
                    # Append the loss series (a column of data) to our list
                    trial_losses.append(df.get_column(TARGET_COLUMN))
                else:
                    print(
                        f"  - Warning: '{TARGET_COLUMN}' not found in {csv_path}. Skipping trial."
                    )
            except Exception as e:
                print(f"  - Error reading {csv_path}: {e}. Skipping trial.")

    if not trial_losses:
        print(
            f"  - No valid trial data found for experiment '{experiment_dir}'. Skipping.\n"
        )
        return None

    print(f"  - Found {len(trial_losses)} trials with valid data.")

    # 2. Find the global min and max loss across all trials for this experiment
    # We concatenate all series into a single one to easily find the overall min/max.
    all_values = pl.concat(trial_losses)
    min_loss = all_values.min()
    max_loss = all_values.max()

    if min_loss is None or max_loss is None:
        print(
            f"  - Could not determine min/max loss for '{experiment_dir}'. Skipping.\n"
        )
        return None

    print(f"  - Loss range for analysis: [{min_loss:.4f}, {max_loss:.4f}]")

    # 3. Create the loss thresholds for the analysis
    # We use np.arange to create evenly spaced thresholds.
    # We add a small epsilon to the max_loss to ensure it's included in the range.
    thresholds = np.arange(min_loss, max_loss + LOSS_STEP, LOSS_STEP)

    # 4. For each threshold, find the number of iterations to converge for each trial
    analysis_results = []
    for threshold in thresholds:
        iters_to_reach_threshold = []
        for loss_series in trial_losses:
            # Find the index of the first value that is less than or equal to the threshold.
            # `arg_true()` returns the indices of all `True` values. `.first()` gets the first one.
            first_index = loss_series.le(threshold).arg_true().first()

            # If `first_index` is not None, it means the trial reached the threshold.
            # The iteration number is the index + 1.
            if first_index is not None:
                iters_to_reach_threshold.append(first_index + 1)

        # 5. Calculate statistics for the current threshold
        num_converged = len(iters_to_reach_threshold)
        total_trials = len(trial_losses)

        if num_converged > 0:
            avg_iters = np.mean(iters_to_reach_threshold)
            std_iters = np.std(iters_to_reach_threshold)
        else:
            # If no trials converged, a Bv_iters is not applicable.
            avg_iters = None
            std_iters = None

        analysis_results.append(
            {
                "case": experiment_dir,
                "loss_threshold": threshold,
                "avg_iters_to_reach": avg_iters,
                "std_iters_to_reach": std_iters,
                "num_converged_trials": num_converged,
                "total_trials": total_trials,
            }
        )

    print(f"  - Analysis complete for {len(thresholds)} thresholds.\n")
    return pl.DataFrame(analysis_results)


def main():
    """
    Main function to run the entire convergence analysis pipeline.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(OUTPUT_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created analysis directory at: {output_dir}")

    if not os.path.isdir(RUNS_DIRECTORY):
        print(
            f"Error: The '{RUNS_DIRECTORY}' directory does not exist. Please run your experiments first."
        )
        return

    all_experiments_df = []
    # Iterate over all directories in the main 'runs' folder
    for experiment_dir in sorted(os.listdir(RUNS_DIRECTORY)):
        if os.path.isdir(os.path.join(RUNS_DIRECTORY, experiment_dir)):
            # Perform the analysis for the current experiment
            experiment_df = perform_convergence_analysis(experiment_dir)
            if experiment_df is not None:
                all_experiments_df.append(experiment_df)

    # 6. Concatenate results from all experiments and save to a single CSV
    if not all_experiments_df:
        print(
            "No experiments were successfully analyzed. No output file will be created."
        )
        return

    final_df = pl.concat(all_experiments_df)

    try:
        final_df.write_csv(OUTPUT_FILE)
        print("=" * 50)
        print(f"✅ Convergence analysis successfully saved to: {OUTPUT_FILE}")
        print("=" * 50)
        print("Final DataFrame preview:")
        print(final_df)
    except Exception as e:
        print(f"Error: Could not write final CSV to {OUTPUT_FILE}. Reason: {e}")


if __name__ == "__main__":
    main()
