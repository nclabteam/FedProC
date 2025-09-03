import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import polars as pl

# Configure Polars display settings
pl.Config.set_tbl_cols(100)
pl.Config.set_tbl_rows(100)

# Import utility functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from analysis.utils import (
    extract_loss_values,
    get_experiment_config,
    get_experiment_paths,
    get_loss_metric_name,
    load_all_experiments,
)


def get_experiment_names_from_excel(excel_path):
    """Read experiment names from the excel file (column '--name=')"""
    df = pl.read_excel(excel_path)
    if "--name=" not in df.columns:
        raise ValueError("Excel file must have a '--name=' column.")
    return set(df["--name="].to_list())


def create_comparison_tables(
    experiment_paths, runs_dir="runs", std_multiplier=1.0, decimal_places=4
):
    """
    Create comparison tables showing test loss across different strategies, grouped by models.

    Args:
        experiment_paths (list): List of experiment directory names
        runs_dir (str): Directory containing experiment folders
        std_multiplier (float): Factor to multiply standard deviation for better visibility
        decimal_places (int): Number of decimal places to display

    Returns:
        tuple: (model_tables, metadata_table) where:
            - model_tables: Dictionary with model names as keys and Polars DataFrames as values
            - metadata_table: DataFrame with dataset, in, out, run_name, experiment info
    """
    # Load experiment data
    all_experiments = []
    for exp_path in experiment_paths:
        exp_dir = os.path.join(runs_dir, exp_path)
        if os.path.isdir(exp_dir) and os.path.exists(
            os.path.join(exp_dir, "results.csv")
        ):
            from analysis.utils import _load_experiment_data

            datum = _load_experiment_data(exp_dir)
            datum["experiment_name"] = exp_path
            all_experiments.append(datum)

    # Group experiments by model
    model_groups = defaultdict(list)
    for exp in all_experiments:
        model_name = exp.get("model", "unknown")
        model_groups[model_name].append(exp)

    # Create metadata table
    metadata_data = []
    model_tables = {}

    for model_name, experiments in model_groups.items():
        combination_data = defaultdict(list)

        for exp in experiments:
            config = get_experiment_config(exp)
            loss_metric = get_loss_metric_name(config["save_local_model"])

            combination_key = (
                model_name,
                config["strategy"],
                config["dataset"],
                config["input_len"],
                config["output_len"],
            )

            for run in exp.get("runs", []):
                run_name = run["name"]

                # Add to metadata table
                metadata_data.append(
                    {
                        "model": model_name,
                        "dataset": config["dataset"],
                        "in": config["input_len"],
                        "out": config["output_len"],
                        "strategy": config["strategy"],
                        "run_name": run_name,
                        "experiment": exp["experiment_name"],
                        "save_local_model": config["save_local_model"],
                        "loss_type": loss_metric,
                    }
                )

                # Extract loss value and collect for statistics
                loss_value = extract_loss_values(run, loss_metric)
                if loss_value is not None and run_name != "avg":
                    combination_data[combination_key].append(loss_value)

        # Calculate statistics for each combination
        table_data = []
        for combination_key, values in combination_data.items():
            model_name, strategy, dataset, input_len, output_len = combination_key

            mean_value = np.mean(values) if values else None
            std_value = np.std(values, ddof=1) if len(values) > 1 else 0.0
            # Apply std multiplier
            std_value *= std_multiplier

            table_data.append(
                {
                    "model": model_name,
                    "strategy": strategy,
                    "dataset": dataset,
                    "in": input_len,
                    "out": output_len,
                    "loss_mean": mean_value,
                    "loss_std": std_value,
                    "n_runs": len(values),
                }
            )

        # Create pivot tables
        if table_data:
            df = pl.DataFrame(table_data)

            mean_pivot = df.pivot(
                values="loss_mean",
                index=["model", "dataset", "in", "out"],
                on="strategy",
            )

            std_pivot = df.pivot(
                values="loss_std",
                index=["model", "dataset", "in", "out"],
                on="strategy",
            )

            model_tables[model_name] = {
                "mean": mean_pivot,
                "std": std_pivot,
                "raw_data": df,
            }

    metadata_table = pl.DataFrame(metadata_data) if metadata_data else None
    return model_tables, metadata_table


def create_ranking_table(model_tables):
    """
    Create ranking tables for each model based on mean performance and std as tiebreaker.

    Args:
        model_tables (dict): Dictionary with model names as keys and tables as values

    Returns:
        dict: Dictionary with model names as keys and ranking DataFrames as values
    """
    ranking_tables = {}

    for model_name, tables in model_tables.items():
        if "mean" not in tables or "std" not in tables:
            continue

        mean_df = tables["mean"]
        std_df = tables["std"]

        # Get strategy columns (exclude index columns)
        strategy_columns = [
            col
            for col in mean_df.columns
            if col not in ["model", "dataset", "in", "out"]
        ]

        if not strategy_columns:
            continue

        ranking_rows = []
        best_strategy_counts = Counter()

        # Process each configuration row
        for i in range(len(mean_df)):
            row_data = {
                "dataset": mean_df["dataset"][i],
                "in": mean_df["in"][i],
                "out": mean_df["out"][i],
            }

            # Get mean and std values for this configuration
            strategy_scores = []
            for strategy in strategy_columns:
                mean_val = mean_df[strategy][i]
                std_val = std_df[strategy][i] if strategy in std_df.columns else 0

                if mean_val is not None and not np.isnan(mean_val):
                    strategy_scores.append((strategy, mean_val, std_val))

            if not strategy_scores:
                continue

            # Sort by mean (ascending - lower is better), then by std (ascending - lower is better)
            strategy_scores.sort(key=lambda x: (x[1], x[2]))

            # Create rankings
            rankings = {}
            for rank, (strategy, mean_val, std_val) in enumerate(strategy_scores, 1):
                rankings[strategy] = rank

            # Add rankings to row
            for strategy in strategy_columns:
                if strategy in rankings:
                    row_data[strategy] = rankings[strategy]
                else:
                    row_data[strategy] = "N/A"

            # Find best strategy (rank 1)
            best_strategies = [s for s, r in rankings.items() if r == 1]
            if best_strategies:
                best_strategy = best_strategies[0]  # In case of ties, take first
                best_strategy_counts[best_strategy] += 1
                row_data["best_strategy"] = best_strategy
            else:
                row_data["best_strategy"] = "N/A"

            ranking_rows.append(row_data)

        # Calculate average ranks
        if ranking_rows:
            avg_ranks = {"dataset": "AVG_RANK", "in": "", "out": ""}
            strategy_avg_ranks = {}

            for strategy in strategy_columns:
                valid_ranks = [
                    row[strategy]
                    for row in ranking_rows
                    if row[strategy] != "N/A"
                    and isinstance(row[strategy], (int, float))
                ]
                if valid_ranks:
                    avg_rank = round(np.mean(valid_ranks), 2)
                    avg_ranks[strategy] = avg_rank
                    strategy_avg_ranks[strategy] = avg_rank
                else:
                    avg_ranks[strategy] = "N/A"
                    strategy_avg_ranks[strategy] = float(
                        "inf"
                    )  # High value for strategies with no data

            # Find most frequent best strategy with tiebreaking by average rank
            if best_strategy_counts:
                max_count = max(best_strategy_counts.values())
                # Get all strategies with the maximum count
                top_strategies = [
                    strategy
                    for strategy, count in best_strategy_counts.items()
                    if count == max_count
                ]

                if len(top_strategies) == 1:
                    # No tie, simple case
                    most_frequent = top_strategies[0]
                    avg_ranks["best_strategy"] = f"{most_frequent} ({max_count}x)"
                else:
                    # Tie in count, use average rank as tiebreaker (lower avg rank wins)
                    best_by_avg_rank = min(
                        top_strategies,
                        key=lambda s: strategy_avg_ranks.get(s, float("inf")),
                    )
                    tied_strategies_info = ", ".join(
                        [
                            f"{s}({best_strategy_counts[s]}x,avg:{strategy_avg_ranks.get(s, 'N/A')})"
                            for s in top_strategies
                        ]
                    )
                    avg_ranks["best_strategy"] = (
                        f"{best_by_avg_rank} (tie: {tied_strategies_info})"
                    )
            else:
                avg_ranks["best_strategy"] = "N/A"

            ranking_rows.append(avg_ranks)

        # Create DataFrame
        if ranking_rows:
            ranking_df = pl.DataFrame(ranking_rows)
            ranking_tables[model_name] = ranking_df

    return ranking_tables


def create_model_specific_tables(
    experiment_paths, runs_dir="runs", std_multiplier=1.0, decimal_places=4
):
    """
    Create individual tables for each model showing mean±std across runs.

    Args:
        experiment_paths (list): List of experiment directory names
        runs_dir (str): Directory containing experiment folders
        std_multiplier (float): Factor to multiply standard deviation for better visibility
        decimal_places (int): Number of decimal places to display

    Returns:
        tuple: (model_tables, metadata_table, ranking_tables) where each model gets its own table
    """
    # Load experiment data
    all_experiments = []
    for exp_path in experiment_paths:
        exp_dir = os.path.join(runs_dir, exp_path)
        if os.path.isdir(exp_dir) and os.path.exists(
            os.path.join(exp_dir, "results.csv")
        ):
            from analysis.utils import _load_experiment_data

            datum = _load_experiment_data(exp_dir)
            datum["experiment_name"] = exp_path
            all_experiments.append(datum)

    # Group experiments by model
    model_groups = defaultdict(list)
    for exp in all_experiments:
        model_name = exp.get("model", "unknown")
        model_groups[model_name].append(exp)

    # Create metadata and model tables
    metadata_data = []
    model_tables = {}
    comparison_tables = {}  # Store mean/std tables for ranking

    for model_name, experiments in model_groups.items():
        if not args.quiet:
            print(f"\nProcessing model: {model_name}")

        combination_data = defaultdict(list)

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
                        "model": model_name,
                        "dataset": config["dataset"],
                        "in": config["input_len"],
                        "out": config["output_len"],
                        "strategy": config["strategy"],
                        "run_name": run_name,
                        "experiment": exp["experiment_name"],
                        "save_local_model": config["save_local_model"],
                        "loss_type": loss_metric,
                    }
                )

                # Collect loss values
                loss_value = extract_loss_values(run, loss_metric)
                if loss_value is not None and run_name != "avg":
                    combination_data[combination_key].append(loss_value)

        # Create table rows for display
        table_rows = []
        config_groups = defaultdict(dict)

        # Data for ranking tables
        table_data = []

        for combination_key, values in combination_data.items():
            strategy, dataset, input_len, output_len = combination_key
            config_key = (dataset, input_len, output_len)

            mean_value = np.mean(values) if values else None
            std_value = np.std(values, ddof=1) if len(values) > 1 else 0.0
            # Apply std multiplier for display only
            std_value_display = std_value * std_multiplier

            config_groups[config_key][strategy] = {
                "mean": mean_value,
                "std": std_value_display,
            }

            # Store raw data for ranking (without multiplier)
            table_data.append(
                {
                    "model": model_name,
                    "strategy": strategy,
                    "dataset": dataset,
                    "in": input_len,
                    "out": output_len,
                    "loss_mean": mean_value,
                    "loss_std": std_value,  # Raw std for ranking
                    "n_runs": len(values),
                }
            )

        # Build table rows for display
        for config_key, strategies in config_groups.items():
            dataset, input_len, output_len = config_key

            row = {"dataset": dataset, "in": input_len, "out": output_len}

            # Add strategy columns with mean±std format using specified decimal places
            for strategy, stats in strategies.items():
                mean_val = stats["mean"]
                std_val = stats["std"]

                if mean_val is not None and std_val is not None:
                    row[strategy] = (
                        f"{mean_val:.{decimal_places}f}±{std_val:.{decimal_places}f}"
                    )
                else:
                    row[strategy] = "N/A"

            table_rows.append(row)

        # Create comparison tables for ranking
        if table_data:
            df = pl.DataFrame(table_data)

            mean_pivot = df.pivot(
                values="loss_mean",
                index=["model", "dataset", "in", "out"],
                on="strategy",
            )

            std_pivot = df.pivot(
                values="loss_std",
                index=["model", "dataset", "in", "out"],
                on="strategy",
            )

            comparison_tables[model_name] = {
                "mean": mean_pivot,
                "std": std_pivot,
            }

        # Create DataFrame for display
        if table_rows:
            model_df = pl.DataFrame(table_rows)
            # Sort by dataset, in, out
            model_df = model_df.sort(["dataset", "in", "out"])
            model_tables[model_name] = model_df
            if not args.quiet:
                print(
                    f"Created table for {model_name} with {len(table_rows)} configurations"
                )

    # Create ranking tables
    ranking_tables = create_ranking_table(comparison_tables)
    # Sort ranking tables as well
    for model_name, ranking_df in ranking_tables.items():
        # Only sort rows that have a dataset value (not the AVG_RANK row)
        data_rows = ranking_df.filter(pl.col("dataset") != "AVG_RANK")
        avg_row = ranking_df.filter(pl.col("dataset") == "AVG_RANK")
        data_rows = data_rows.sort(["dataset", "in", "out"])
        ranking_tables[model_name] = pl.concat([data_rows, avg_row])

    metadata_table = pl.DataFrame(metadata_data) if metadata_data else None
    return model_tables, metadata_table, ranking_tables


def display_comparison_tables(
    model_tables, metadata_table=None, std_multiplier=1.0, decimal_places=4
):
    """Display comparison tables with mean and std."""
    for model_name, tables in model_tables.items():
        print(f"\n{'='*50}")
        print(f"MODEL: {model_name.upper()} - MEAN VALUES")
        print(f"{'='*50}")
        print(tables["mean"])

        print(f"\n{'='*50}")
        print(f"MODEL: {model_name.upper()} - STANDARD DEVIATION")
        if std_multiplier != 1.0:
            print(f"(multiplied by {std_multiplier})")
        print(f"{'='*50}")
        print(tables["std"])
        print()

    if metadata_table is not None:
        print(f"\n{'='*60}")
        print("METADATA TABLE (Experiment Details)")
        print(f"{'='*60}")
        print(metadata_table)


def display_model_tables(
    model_tables,
    ranking_tables=None,
    metadata_table=None,
    show_metadata=True,
    std_multiplier=1.0,
    decimal_places=4,
):
    """Display individual model tables with rankings."""
    for model_name, table in model_tables.items():
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name.upper()}")
        if std_multiplier != 1.0:
            print(f"(Standard deviation multiplied by {std_multiplier})")
        if decimal_places != 4:
            print(f"(Displayed with {decimal_places} decimal places)")
        print(f"{'='*80}")
        print("Each row shows a unique configuration (dataset, in, out)")
        print("Columns show mean±std across multiple runs for each strategy")
        print("-" * 80)
        print(table)
        print()

        # Display ranking table
        if ranking_tables and model_name in ranking_tables:
            print(f"\n{'='*80}")
            print(f"RANKING TABLE FOR {model_name.upper()}")
            print(f"{'='*80}")
            print(
                "Strategies ranked by mean performance (1=best, lower loss is better)"
            )
            print("Ties broken by standard deviation (lower std is better)")
            print("Last row shows average rank across all configurations")
            print("Last column shows which strategy wins most often")
            print("-" * 80)
            print(ranking_tables[model_name])
            print()

    if metadata_table is not None and show_metadata:
        print(f"\n{'='*80}")
        print("METADATA TABLE (All Runs and Experiments)")
        print(f"{'='*80}")
        print(metadata_table)


def save_comparison_tables(
    model_tables,
    metadata_table=None,
    output_dir="analysis/tables",
    std_multiplier=1.0,
    decimal_places=4,
):
    """Save comparison tables to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    # Add std_multiplier and decimal_places info to filenames if not default
    suffix = ""
    if std_multiplier != 10000:
        suffix += f"_stdx{std_multiplier}"
    if decimal_places != 3:
        suffix += f"_dec{decimal_places}"

    for model_name, tables in model_tables.items():
        # Save mean table
        mean_filepath = os.path.join(
            output_dir, f"{model_name}_mean_comparison{suffix}.csv"
        )
        tables["mean"].write_csv(mean_filepath)
        print(f"Saved mean table for {model_name} to {mean_filepath}")

        # Save std table
        std_filepath = os.path.join(
            output_dir, f"{model_name}_std_comparison{suffix}.csv"
        )
        tables["std"].write_csv(std_filepath)
        print(f"Saved std table for {model_name} to {std_filepath}")

        # Save raw data
        raw_filepath = os.path.join(output_dir, f"{model_name}_raw_data{suffix}.csv")
        tables["raw_data"].write_csv(raw_filepath)
        print(f"Saved raw data for {model_name} to {raw_filepath}")

    if metadata_table is not None:
        metadata_filepath = os.path.join(output_dir, f"experiment_metadata{suffix}.csv")
        metadata_table.write_csv(metadata_filepath)
        print(f"Saved metadata table to {metadata_filepath}")


def save_model_tables(
    model_tables,
    ranking_tables=None,
    metadata_table=None,
    output_dir="analysis/tables",
    std_multiplier=1.0,
    decimal_places=4,
):
    """Save individual model tables and ranking tables to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    # Add std_multiplier and decimal_places info to filenames if not default
    suffix = ""
    if std_multiplier != 10000:
        suffix += f"_stdx{std_multiplier}"
    if decimal_places != 3:
        suffix += f"_dec{decimal_places}"

    for model_name, table in model_tables.items():
        # Save analysis table
        filename = f"{model_name}_analysis{suffix}.csv"
        filepath = os.path.join(output_dir, filename)
        table.write_csv(filepath)
        print(f"Saved analysis table for {model_name} to {filepath}")

        # Save ranking table
        if ranking_tables and model_name in ranking_tables:
            ranking_filename = f"{model_name}_ranking{suffix}.csv"
            ranking_filepath = os.path.join(output_dir, ranking_filename)
            ranking_tables[model_name].write_csv(ranking_filepath)
            print(f"Saved ranking table for {model_name} to {ranking_filepath}")

    if metadata_table is not None:
        metadata_filepath = os.path.join(output_dir, f"experiment_metadata{suffix}.csv")
        metadata_table.write_csv(metadata_filepath)
        print(f"Saved metadata table to {metadata_filepath}")


def create_summary_table(model_tables, decimal_places=4):
    """Create summary tables showing average performance across all models."""
    mean_summary_data = []
    std_summary_data = []

    for model_name, tables in model_tables.items():
        strategy_columns = [
            col
            for col in tables["mean"].columns
            if col not in ["model", "dataset", "in", "out"]
        ]

        mean_summary_row = {"model": model_name}
        std_summary_row = {"model": model_name}

        for strategy in strategy_columns:
            mean_value = tables["mean"].select(pl.col(strategy).mean()).item()
            std_value = tables["std"].select(pl.col(strategy).mean()).item()

            mean_summary_row[strategy] = mean_value
            std_summary_row[strategy] = std_value

        mean_summary_data.append(mean_summary_row)
        std_summary_data.append(std_summary_row)

    mean_summary = pl.DataFrame(mean_summary_data) if mean_summary_data else None
    std_summary = pl.DataFrame(std_summary_data) if std_summary_data else None

    return mean_summary, std_summary


def create_combined_summary_table(model_tables, decimal_places=4):
    """Create combined summary table with mean±std format."""
    summary_data = []

    for model_name, tables in model_tables.items():
        strategy_columns = [
            col
            for col in tables["mean"].columns
            if col not in ["model", "dataset", "in", "out"]
        ]

        summary_row = {"model": model_name}

        for strategy in strategy_columns:
            mean_value = tables["mean"].select(pl.col(strategy).mean()).item()
            std_value = tables["std"].select(pl.col(strategy).mean()).item()

            if mean_value is not None and std_value is not None:
                summary_row[strategy] = (
                    f"{mean_value:.{decimal_places}f}±{std_value:.{decimal_places}f}"
                )
            else:
                summary_row[strategy] = "N/A"

        summary_data.append(summary_row)

    return pl.DataFrame(summary_data) if summary_data else None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate analysis tables from federated learning experiment results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/Output arguments
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

    # Analysis type arguments
    parser.add_argument(
        "--table-type",
        "-t",
        choices=["model-specific", "comparison", "both"],
        default="model-specific",
        help="Type of tables to generate",
    )

    # Formatting arguments
    parser.add_argument(
        "--std-multiplier",
        "-s",
        type=float,
        default=10e3,
        help="Factor to multiply standard deviation for better visibility (e.g., 100 for percentage-like display)",
    )

    parser.add_argument(
        "--decimal-places",
        "-d",
        type=int,
        default=3,
        help="Number of decimal places to display in the results",
    )

    # Display arguments
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

    # Filtering arguments
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

    # Experiment selection
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        help="Process only specific experiments (e.g., --experiments exp76 exp77)",
    )

    # Excel experiment filter
    parser.add_argument(
        "--excel",
        type=str,
        help="Excel file to filter experiments (column '--name=' should contain experiment names)",
    )

    return parser.parse_args()


def filter_experiments(experiments, args):
    """Filter experiments based on command line arguments."""
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


def main():
    """Main function with command line argument handling."""
    global args
    args = parse_args()

    if not args.quiet:
        print(f"Loading experiments from: {args.runs_dir}")
        print(f"Standard deviation will be multiplied by: {args.std_multiplier}")
        print(f"Results will be displayed with {args.decimal_places} decimal places")

    # Load experiments using the utils function with custom runs_dir
    from analysis.utils import load_all_experiments

    experiments = load_all_experiments(runs_dir=args.runs_dir)

    if not experiments:
        print(f"No valid experiments found in {args.runs_dir}")
        return

    # Filter experiments if requested
    experiments = filter_experiments(experiments, args)

    # Additional filter: Excel file
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

    # Generate tables based on type
    if args.table_type in ["model-specific", "both"]:
        if not args.quiet:
            print("\nGenerating model-specific tables...")

        model_tables, metadata_table, ranking_tables = create_model_specific_tables(
            experiment_paths,
            runs_dir=args.runs_dir,
            std_multiplier=args.std_multiplier,
            decimal_places=args.decimal_places,
        )

        if not args.no_display:
            display_model_tables(
                model_tables,
                ranking_tables,
                metadata_table if args.show_metadata else None,
                show_metadata=args.show_metadata,
                std_multiplier=args.std_multiplier,
                decimal_places=args.decimal_places,
            )

        # Save tables (metadata always saved)
        save_model_tables(
            model_tables,
            ranking_tables,
            metadata_table,  # Always save metadata
            output_dir=args.output_dir,
            std_multiplier=args.std_multiplier,
            decimal_places=args.decimal_places,
        )

    if args.table_type in ["comparison", "both"]:
        if not args.quiet:
            print("\nGenerating comparison tables...")

        comp_tables, comp_metadata = create_comparison_tables(
            experiment_paths,
            runs_dir=args.runs_dir,
            std_multiplier=args.std_multiplier,
            decimal_places=args.decimal_places,
        )

        if not args.no_display:
            display_comparison_tables(
                comp_tables,
                comp_metadata if args.show_metadata else None,
                std_multiplier=args.std_multiplier,
                decimal_places=args.decimal_places,
            )

        # Save comparison tables (metadata always saved)
        save_comparison_tables(
            comp_tables,
            comp_metadata,  # Always save metadata
            output_dir=args.output_dir,
            std_multiplier=args.std_multiplier,
            decimal_places=args.decimal_places,
        )

    if not args.quiet:
        print(f"\nTables saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
