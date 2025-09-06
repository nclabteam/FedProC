import argparse
import os
import sys

import polars as pl

# Configure Polars display settings
pl.Config.set_tbl_cols(100)
pl.Config.set_tbl_rows(100)

# Import utility functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.analysis import (
    create_ranking_table_from_pivot,
    extract_loss_values,
    filter_experiments,
    get_experiment_config,
    get_experiment_names_from_excel,
    get_experiment_paths,
    get_loss_metric_name,
    load_all_experiments,
    parse_args,
)


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
                model_name,
                config["strategy"],
                config["dataset"],
                config["input_len"],
                config["output_len"],
            )
            for run in exp.get("runs", []):
                run_name = run["name"]
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
                loss_value = extract_loss_values(run, loss_metric)
                if loss_value is not None and run_name != "avg":
                    combination_data.setdefault(combination_key, []).append(loss_value)

        table_data = []
        for combination_key, values in combination_data.items():
            model_name, strategy, dataset, input_len, output_len = combination_key
            mean_value = pl.Series(values).mean() if values else None
            std_value = pl.Series(values).std() if len(values) > 1 else 0.0
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
            ).sort(["model", "dataset", "in", "out"])

            std_pivot = df.pivot(
                values="loss_std",
                index=["model", "dataset", "in", "out"],
                on="strategy",
            ).sort(["model", "dataset", "in", "out"])

            model_tables[model_name] = {
                "mean": mean_pivot,
                "std": std_pivot,
                "raw_data": df,
            }

    metadata_table = pl.DataFrame(metadata_data) if metadata_data else None
    return model_tables, metadata_table


def create_ranking_table(model_tables, decimal_places=4, std_multiplier=1.0):
    """
    Create ranking tables for each model based on rounded mean performance and rounded std as tiebreaker.

    Args:
        model_tables (dict): Dictionary with model names as keys and tables as values
        decimal_places (int): Number of decimal places to round before ranking

    Returns:
        dict: Dictionary with model names as keys and ranking DataFrames as values
    """
    ranking_tables = {}

    for model_name, tables in model_tables.items():
        if "mean" not in tables or "std" not in tables:
            continue

        ranking_df = create_ranking_table_from_pivot(
            main_df=tables["mean"],
            tiebreak_df=tables["std"],
            decimal_places=decimal_places,
            sort_cols=["dataset", "in", "out"],
        )
        if ranking_df is not None:
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
            from utils.analysis import _load_experiment_data

            datum = _load_experiment_data(exp_dir)
            datum["experiment_name"] = exp_path
            all_experiments.append(datum)

    # Group experiments by model
    model_groups = {}
    for exp in all_experiments:
        model_name = exp.get("model", "unknown")
        model_groups.setdefault(model_name, []).append(exp)

    # Create metadata and model tables
    metadata_data = []
    model_tables = {}
    comparison_tables = {}  # Store mean/std tables for ranking

    for model_name, experiments in model_groups.items():
        if not args.quiet:
            print(f"\nProcessing model: {model_name}")

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
                    combination_data.setdefault(combination_key, []).append(loss_value)

        # Create table rows for display
        table_rows = []
        config_groups = {}

        # Data for ranking tables
        table_data = []

        for combination_key, values in combination_data.items():
            strategy, dataset, input_len, output_len = combination_key
            config_key = (dataset, input_len, output_len)

            mean_value = pl.Series(values).mean() if values else None
            std_value = pl.Series(values).std() if len(values) > 1 else 0.0
            # Apply std multiplier for display only
            std_value_display = std_value * std_multiplier

            config_groups.setdefault(config_key, {})[strategy] = {
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
                    "loss_std": std_value_display,
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

                # Multiply std by std_multiplier before rounding and displaying
                if mean_val is not None and std_val is not None:
                    std_val_display = round(std_val, decimal_places)
                    row[
                        strategy
                    ] = f"{mean_val:.{decimal_places}f}±{std_val_display:.{decimal_places}f}"
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
    ranking_tables = create_ranking_table(
        model_tables=comparison_tables,
        decimal_places=decimal_places,
        std_multiplier=std_multiplier,
    )
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
        print(f"(Standard deviation multiplied by {std_multiplier})")
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
                summary_row[
                    strategy
                ] = f"{mean_value:.{decimal_places}f}±{std_value:.{decimal_places}f}"
            else:
                summary_row[strategy] = "N/A"

        summary_data.append(summary_row)

    return pl.DataFrame(summary_data) if summary_data else None


def main():
    """Main function with command line argument handling."""
    global args
    args = parse_args(default_table_type="model-specific")

    if not args.quiet:
        print(f"Loading experiments from: {args.runs_dir}")
        print(f"Standard deviation will be multiplied by: {args.std_multiplier}")
        print(f"Results will be displayed with {args.decimal_places} decimal places")

    # Load experiments using the utils function with custom runs_dir
    from utils.analysis import load_all_experiments

    experiments = load_all_experiments(runs_dir=args.runs_dir)
    if not experiments:
        print(f"No valid experiments found in {args.runs_dir}")
        return
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
