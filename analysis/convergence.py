import json
import os

import numpy as np
import polars as pl

# --- Configuration ---
# The column to analyze for convergence.
TARGET_COLUMN = "global_avg_test_loss"
# The step size for creating loss thresholds.
LOSS_STEP = 0.0001
# The number of decimal places to round floats to.
ROUND_PRECISION = 4
# The directory containing all experimental runs.
RUNS_DIRECTORY = "runs"
# The output file for the analysis.
OUTPUT_FILE = os.path.join("analysis", "convergence.csv")


def perform_convergence_analysis(experiment_dir: str):
    """
    Analyzes the convergence for all trials within a single experiment directory.
    """
    print(f"--- Analyzing Experiment: {experiment_dir} ---")
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

    # 1. Load the loss data from each trial into a list of Polars Series (`dfs` in your description)
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
    print(f"  - Found {len(trial_losses)} trials with valid data.")

    # --- NEW: Manual implementation of `(sum(dfs) / len(dfs))` ---
    print("  - Manually calculating mean learning curve to determine analysis range...")

    # 2a. Find the length of the longest trial to define the size of our arrays.
    max_len = 0
    for s in trial_losses:
        if len(s) > max_len:
            max_len = len(s)

    # 2b. Initialize arrays for summation and counting.
    sum_at_iteration = np.zeros(max_len, dtype=np.float64)
    count_at_iteration = np.zeros(max_len, dtype=np.int32)

    # 2c. Loop through each trial and add its values to the sum and increment the count.
    for loss_series in trial_losses:
        # Convert series to numpy array for efficient numeric operations
        values = loss_series.to_numpy()
        series_len = len(values)
        # Add this trial's values to the running sum for the iterations it completed
        sum_at_iteration[:series_len] += values
        # Increment the count for the iterations this trial completed
        count_at_iteration[:series_len] += 1

    # 2d. Calculate the mean curve, safely handling division by zero for any trailing iterations.
    mean_loss_curve_np = np.divide(
        sum_at_iteration,
        count_at_iteration,
        out=np.full(max_len, np.nan),  # Put NaN where count is 0
        where=count_at_iteration != 0,
    )
    # Convert the final numpy array back to a Polars Series
    mean_loss_curve = pl.Series(values=mean_loss_curve_np).drop_nans()
    # --- END OF MANUAL CALCULATION BLOCK ---

    # 3. Find min/max FROM THE MEAN CURVE and round them.
    min_loss = round(mean_loss_curve.min(), ROUND_PRECISION)
    max_loss = round(mean_loss_curve.max(), ROUND_PRECISION)

    if min_loss is None or max_loss is None:
        print(
            f"  - Could not determine min/max loss from mean curve for '{experiment_dir}'. Skipping.\n"
        )
        return None
    print(f"  - Rounded loss range (from mean curve): [{min_loss:.4f}, {max_loss:.4f}]")

    # 4. Create the loss thresholds for the analysis
    thresholds = np.arange(min_loss, max_loss + LOSS_STEP, LOSS_STEP)

    # 5. For each threshold, find the number of iterations to converge for each original trial
    analysis_results = []
    for threshold in thresholds:
        iters_to_reach_threshold = []
        for loss_series in trial_losses:
            first_index = loss_series.le(threshold).arg_true().first()
            if first_index is not None:
                iters_to_reach_threshold.append(first_index + 1)

        num_converged = len(iters_to_reach_threshold)
        avg_iters, std_iters = (
            (np.mean(iters_to_reach_threshold), np.std(iters_to_reach_threshold))
            if num_converged > 0
            else (None, None)
        )

        analysis_results.append(
            {
                "case": experiment_dir,
                **metadata,
                "loss_threshold": threshold,
                "avg_iters_to_reach": avg_iters,
                "std_iters_to_reach": std_iters,
                "num_converged_trials": num_converged,
                "total_trials": len(trial_losses),
            }
        )

    print(f"  - Analysis complete for {len(thresholds)} thresholds.\n")
    return pl.DataFrame(analysis_results)


def main():
    """Main function to run the entire convergence analysis pipeline."""
    output_dir = os.path.dirname(OUTPUT_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created analysis directory at: {output_dir}")

    if not os.path.isdir(RUNS_DIRECTORY):
        print(f"Error: The '{RUNS_DIRECTORY}' directory does not exist.")
        return

    all_experiments_df = []
    for experiment_dir in sorted(os.listdir(RUNS_DIRECTORY)):
        if os.path.isdir(os.path.join(RUNS_DIRECTORY, experiment_dir)):
            experiment_df = perform_convergence_analysis(experiment_dir)
            if experiment_df is not None:
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
        print(f"✅ Convergence analysis successfully saved to: {OUTPUT_FILE}")
        print("=" * 50)
        print("Final DataFrame preview:")
        print(final_df)
    except Exception as e:
        print(f"Error: Could not write final CSV to {OUTPUT_FILE}. Reason: {e}")


if __name__ == "__main__":
    main()
