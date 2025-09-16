import os
import sys
from collections import Counter

import numpy as np
import polars as pl

# Configure Polars display settings
pl.Config.set_tbl_cols(100)
pl.Config.set_tbl_rows(100)

# Import utility functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.analysis import (
    filter_experiments,
    get_experiment_config,
    get_experiment_names_from_excel,
    get_experiment_paths,
    get_loss_metric_name,
    load_all_experiments,
    parse_args,
)


def analyze_convergence_stability(loss_sequence, improvement_threshold=0.0):
    """
    Analyze convergence and stability metrics from a loss sequence.

    Args:
        loss_sequence (list): Sequence of loss values over training rounds
        improvement_threshold (float): Minimum improvement to consider as meaningful

    Returns:
        dict: Dictionary containing stability metrics
    """
    if not loss_sequence or len(loss_sequence) < 2:
        return {
            "lowest_loss_round": None,
            "longest_improvement_streak": 0,
            "most_frequent_improvement_streak": 0,
            "total_improvement_rounds": 0,
            "oscillation_count": 0,
            "improvement_ratio": 0.0,
        }

    # Calculate improvements (negative means improvement since we want lower loss)
    improvements = []
    improvement_streaks = []
    current_streak = 0
    oscillation_count = 0
    total_improvement_rounds = 0

    # Find the round with the lowest loss
    lowest_loss_value = min(loss_sequence)
    lowest_loss_round = loss_sequence.index(lowest_loss_value)

    for i in range(1, len(loss_sequence)):
        improvement = loss_sequence[i - 1] - loss_sequence[i]  # Positive = improvement
        improvements.append(improvement)

        if improvement > improvement_threshold:
            # Meaningful improvement
            current_streak += 1
            total_improvement_rounds += 1
        else:
            # No improvement or degradation
            if current_streak > 0:
                improvement_streaks.append(current_streak)
                current_streak = 0

            # Count oscillations (when loss increases after previous decrease)
            if improvement < -improvement_threshold and i > 1:
                prev_improvement = improvements[i - 2] if i > 1 else 0
                if prev_improvement > improvement_threshold:
                    oscillation_count += 1

    # Add final streak if training ended during improvement
    if current_streak > 0:
        improvement_streaks.append(current_streak)

    # Calculate metrics
    longest_improvement_streak = max(improvement_streaks) if improvement_streaks else 0

    # Most frequent streak length
    if improvement_streaks:
        streak_counts = Counter(improvement_streaks)
        most_frequent_improvement_streak = streak_counts.most_common(1)[0][0]
    else:
        most_frequent_improvement_streak = 0

    # Improvement ratio
    improvement_ratio = (
        total_improvement_rounds / (len(loss_sequence) - 1)
        if len(loss_sequence) > 1
        else 0.0
    )

    return {
        "lowest_loss_round": lowest_loss_round,
        "longest_improvement_streak": longest_improvement_streak,
        "most_frequent_improvement_streak": most_frequent_improvement_streak,
        "total_improvement_rounds": total_improvement_rounds,
        "oscillation_count": oscillation_count,
        "improvement_ratio": improvement_ratio,
    }


def create_stability_tables(
    experiment_paths, runs_dir="runs", decimal_places=3, improvement_threshold=1e-6
):
    """
    Create stability analysis tables for experiments.

    Args:
        experiment_paths (list): List of experiment directory names
        runs_dir (str): Directory containing experiment folders
        decimal_places (int): Number of decimal places to display
        improvement_threshold (float): Minimum improvement threshold

    Returns:
        tuple: (model_tables, metadata_table)
    """
    # Load experiment data
    all_experiments = []
    for exp_path in experiment_paths:
        exp_dir = os.path.join(runs_dir, exp_path)
        if os.path.isdir(exp_dir) and os.path.exists(
            os.path.join(exp_dir, "results.csv")
        ):
            from utils.analysis import _load_experiment_data

            datum = _load_experiment_data(exp_dir)
            datum["experiment_name"] = exp_path
            all_experiments.append(datum)

    # Group experiments by model
    model_groups = {}
    for exp in all_experiments:
        model_name = exp.get("model", "unknown")
        model_groups.setdefault(model_name, []).append(exp)

    metadata_data = []
    model_tables = {}

    for model_name, experiments in model_groups.items():
        combination_data = {}

        for exp in experiments:
            config = get_experiment_config(exp)
            loss_metric = get_loss_metric_name(config["save_local_model"])

            combination_key = (
                config["strategy"],
                config["dataset"],
                config["input_len"],
                config["output_len"],
            )

            for run in exp.get("runs", []):
                run_name = run["name"]

                # Add to metadata
                metadata_data.append(
                    {
                        "dataset": config["dataset"],
                        "in": config["input_len"],
                        "out": config["output_len"],
                        "strategy": config["strategy"],
                        "run_name": run_name,
                        "experiment": exp["experiment_name"],
                        "loss_metric": loss_metric,
                    }
                )

                # Get full loss sequence (not aggregated)
                if loss_metric in run and run_name != "avg":
                    loss_sequence = run[loss_metric]
                    if isinstance(loss_sequence, list) and len(loss_sequence) > 1:
                        stability_metrics = analyze_convergence_stability(
                            loss_sequence=loss_sequence,
                            improvement_threshold=improvement_threshold,
                        )
                        combination_data.setdefault(combination_key, []).append(
                            stability_metrics
                        )

        # Aggregate metrics across runs
        table_data = []
        for combination_key, metrics_list in combination_data.items():
            strategy, dataset, input_len, output_len = combination_key

            if not metrics_list:
                continue

            # Calculate mean and std for each metric
            aggregated_metrics = {}
            for metric_name in metrics_list[0].keys():
                values = [
                    m[metric_name] for m in metrics_list if m[metric_name] is not None
                ]
                if values:
                    aggregated_metrics[f"{metric_name}_mean"] = round(
                        np.mean(values), decimal_places
                    )
                    aggregated_metrics[f"{metric_name}_std"] = (
                        round(np.std(values), decimal_places)
                        if len(values) > 1
                        else 0.0
                    )
                else:
                    aggregated_metrics[f"{metric_name}_mean"] = None
                    aggregated_metrics[f"{metric_name}_std"] = 0.0

            table_data.append(
                {
                    "strategy": strategy,
                    "dataset": dataset,
                    "in": input_len,
                    "out": output_len,
                    "n_runs": len(metrics_list),
                    **aggregated_metrics,
                }
            )

        if table_data:
            df = pl.DataFrame(table_data)

            # Create pivot tables for different metrics
            metrics_to_pivot = [
                "lowest_loss_round",
                "longest_improvement_streak",
                "most_frequent_improvement_streak",
                "oscillation_count",
                "improvement_ratio",
            ]

            pivot_tables = {}
            for metric in metrics_to_pivot:
                mean_col = f"{metric}_mean"
                std_col = f"{metric}_std"

                if mean_col in df.columns:
                    mean_pivot = df.pivot(
                        values=mean_col,
                        index=["dataset", "in", "out"],
                        on="strategy",
                    ).sort(["dataset", "in", "out"])

                    std_pivot = df.pivot(
                        values=std_col,
                        index=["dataset", "in", "out"],
                        on="strategy",
                    ).sort(["dataset", "in", "out"])

                    pivot_tables[metric] = {"mean": mean_pivot, "std": std_pivot}

            model_tables[model_name] = {"raw_data": df, **pivot_tables}

    metadata_table = pl.DataFrame(metadata_data) if metadata_data else None
    return model_tables, metadata_table


def display_stability_tables(
    model_tables,
    metadata_table=None,
    show_metadata=True,
    decimal_places=3,
    improvement_threshold=1e-6,
):
    """Display stability analysis tables with detailed explanations."""
    metric_descriptions = {
        "lowest_loss_round": {
            "title": "Round with the lowest loss achieved (best performance point)",
            "explanation": "Shows the training round where the model achieved its lowest loss value.\nLower values = Model peaked early in training\nHigher values = Model continued improving throughout training\nIndicates the optimal stopping point for best model performance.",
        },
        "longest_improvement_streak": {
            "title": "Maximum consecutive rounds of improvement (stability measure)",
            "explanation": "The longest sequence of consecutive rounds with meaningful loss reduction.\nLower values = Erratic training with frequent plateaus or oscillations\nHigher values = Stable, consistent optimization progress\nReflects training smoothness and optimization algorithm effectiveness.",
        },
        "most_frequent_improvement_streak": {
            "title": "Most common improvement streak length (training pattern)",
            "explanation": "The streak length that occurred most often during training.\nLower values = Training progresses in short bursts with frequent stagnation\nHigher values = Sustained improvement patterns dominate the training\nReveals the typical learning rhythm and optimization behavior.",
        },
        "oscillation_count": {
            "title": "Number of loss increases after decreases (instability measure)",
            "explanation": f"Counts how many times loss went up by more than {improvement_threshold} (threshold) after going down.\nLower values = Stable, monotonic improvement (ideal)\nHigher values = Unstable training, possible learning rate or optimization issues\nHigh oscillations suggest need for learning rate adjustment or different optimizer.",
        },
        "improvement_ratio": {
            "title": "Fraction of rounds with improvement (training efficiency: 0.0-1.0)",
            "explanation": f"Proportion of training rounds that resulted in loss reduction > {improvement_threshold} (threshold).\nLower values = Inefficient training, many wasted rounds\nHigher values = Efficient training, most rounds contributed to learning\nValues >0.5 indicate productive training; <0.3 suggests optimization problems.",
        },
    }

    for model_name, tables in model_tables.items():
        print(f"\n{'='*80}")
        print(f"STABILITY ANALYSIS FOR MODEL: {model_name.upper()}")
        print(f"Improvement threshold: {improvement_threshold}")
        print(f"{'='*80}")

        # Display each metric table
        for metric, info in metric_descriptions.items():
            if metric in tables and "mean" in tables[metric]:
                print(f"\n{info['title']}")
                print(f"{'-'*60}")
                print(info["explanation"])
                print(f"{'-'*60}")

                # Combine mean and std for display
                mean_df = tables[metric]["mean"]
                std_df = tables[metric]["std"]

                # Create display table with mean±std format
                display_df = mean_df.clone()
                for col in mean_df.columns:
                    if col in ["dataset", "in", "out"]:
                        continue
                    if col in std_df.columns:
                        mean_col = mean_df[col].round(decimal_places)
                        std_col = std_df[col].round(decimal_places)
                        display_df = display_df.with_columns(
                            [(mean_col.cast(str) + "±" + std_col.cast(str)).alias(col)]
                        )
                    else:
                        display_df = display_df.with_columns([pl.col(col).cast(str)])

                print(display_df)
                print()

    if metadata_table is not None and show_metadata:
        print(f"\n{'='*80}")
        print("METADATA TABLE (All Runs and Experiments)")
        print(f"{'='*80}")
        print(metadata_table)

    print(f"\n{'='*100}")


def save_stability_tables(
    model_tables,
    metadata_table=None,
    output_dir="analysis/tables",
    decimal_places=3,
):
    """Save stability analysis tables to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    suffix = f"_stability_dec{decimal_places}" if decimal_places != 3 else "_stability"

    for model_name, tables in model_tables.items():
        # Save raw data
        raw_filepath = os.path.join(
            output_dir, f"{model_name}_stability_raw{suffix}.csv"
        )
        tables["raw_data"].write_csv(raw_filepath)
        print(f"Saved stability raw data for {model_name} to {raw_filepath}")

        # Save individual metric tables
        for metric in tables:
            if metric != "raw_data" and isinstance(tables[metric], dict):
                if "mean" in tables[metric]:
                    mean_filepath = os.path.join(
                        output_dir, f"{model_name}_{metric}_mean{suffix}.csv"
                    )
                    tables[metric]["mean"].write_csv(mean_filepath)

                if "std" in tables[metric]:
                    std_filepath = os.path.join(
                        output_dir, f"{model_name}_{metric}_std{suffix}.csv"
                    )
                    tables[metric]["std"].write_csv(std_filepath)

        print(f"Saved stability metric tables for {model_name}")

    if metadata_table is not None:
        metadata_filepath = os.path.join(output_dir, f"stability_metadata{suffix}.csv")
        metadata_table.write_csv(metadata_filepath)
        print(f"Saved stability metadata table to {metadata_filepath}")


def main():
    """Main function with command line argument handling."""
    # Parse arguments normally
    args = parse_args(default_table_type="model-specific")

    # Add improvement_threshold as a custom attribute if not already present
    if not hasattr(args, "improvement_threshold"):
        args.improvement_threshold = 1e-6

    if not args.quiet:
        print(f"Loading experiments from: {args.runs_dir}")
        print(f"Results will be displayed with {args.decimal_places} decimal places")
        print(f"Using improvement threshold: {args.improvement_threshold}")

    experiments = load_all_experiments(runs_dir=args.runs_dir)
    if not experiments:
        print(f"No valid experiments found in {args.runs_dir}")
        return

    experiments = filter_experiments(experiments, args)

    if args.excel:
        if not os.path.exists(args.excel):
            print(f"Excel file not found: {args.excel}")
            return
        excel_names = get_experiment_names_from_excel(args.excel)
        experiments = [
            exp for exp in experiments if exp.get("experiment_name", "") in excel_names
        ]
        if not args.quiet:
            print(f"Filtered experiments using Excel file: {args.excel}")
            print(f"Remaining experiments: {len(experiments)}")

    if not experiments:
        print("No experiments match the specified filters")
        return

    experiment_paths = get_experiment_paths(experiments)

    if not args.quiet:
        print(f"Processing {len(experiment_paths)} experiments...")
        if args.models:
            print(f"  Models: {args.models}")
        if args.strategies:
            print(f"  Strategies: {args.strategies}")
        if args.datasets:
            print(f"  Datasets: {args.datasets}")

    # Create stability tables
    model_tables, metadata_table = create_stability_tables(
        experiment_paths,
        runs_dir=args.runs_dir,
        decimal_places=args.decimal_places,
        improvement_threshold=args.improvement_threshold,
    )

    if not args.no_display:
        display_stability_tables(
            model_tables=model_tables,
            metadata_table=metadata_table if args.show_metadata else None,
            show_metadata=args.show_metadata,
            decimal_places=args.decimal_places,
            improvement_threshold=args.improvement_threshold,
        )

    save_stability_tables(
        model_tables=model_tables,
        metadata_table=metadata_table,
        output_dir=args.output_dir,
        decimal_places=args.decimal_places,
    )

    if not args.quiet:
        print(f"\nStability analysis tables saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
