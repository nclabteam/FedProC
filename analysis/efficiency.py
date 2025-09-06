import argparse
import os
import sys

import polars as pl

pl.Config.set_tbl_cols(100)
pl.Config.set_tbl_rows(100)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.analysis import (
    create_ranking_table_from_pivot,
    extract_loss_values,
    filter_experiments,
    get_experiment_config,
    get_experiment_names_from_excel,
    get_experiment_paths,
    load_all_experiments,
    parse_args,
)


def create_efficiency_tables(
    experiment_paths, runs_dir="runs", decimal_places=3, std_multiplier=1.0
):
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
            combination_key = (
                config["strategy"],
                config["dataset"],
                config["input_len"],
                config["output_len"],
            )
            for run in exp.get("runs", []):
                run_name = run["name"]
                metadata_data.append(
                    {
                        "dataset": config["dataset"],
                        "in": config["input_len"],
                        "out": config["output_len"],
                        "strategy": config["strategy"],
                        "run_name": run_name,
                        "experiment": exp["experiment_name"],
                    }
                )
                time_values = extract_loss_values(run, "time_per_iter")
                if time_values is not None and run_name != "avg":
                    if (
                        isinstance(time_values, (list, tuple, pl.Series))
                        and len(time_values) > 0
                    ):
                        total_time = float(pl.Series(time_values).sum())
                        avg_time = float(pl.Series(time_values).mean())
                    else:
                        total_time = float(time_values)
                        avg_time = float(time_values)
                    combination_data.setdefault(combination_key, []).append(
                        (total_time, avg_time)
                    )

        table_data = []
        for combination_key, values in combination_data.items():
            strategy, dataset, input_len, output_len = combination_key
            total_times = [v[0] for v in values]
            avg_times = [v[1] for v in values]
            mean_total = pl.Series(total_times).mean() if total_times else None
            std_total = pl.Series(total_times).std() if len(total_times) > 1 else 0.0
            mean_avg = pl.Series(avg_times).mean() if avg_times else None
            std_avg = pl.Series(avg_times).std() if len(avg_times) > 1 else 0.0

            table_data.append(
                {
                    "strategy": strategy,
                    "dataset": dataset,
                    "in": input_len,
                    "out": output_len,
                    "total_time_mean": (
                        round(mean_total, decimal_places)
                        if mean_total is not None
                        else None
                    ),
                    "total_time_std": round(std_total, decimal_places),
                    "avg_time_mean": (
                        round(mean_avg, decimal_places)
                        if mean_avg is not None
                        else None
                    ),
                    "avg_time_std": round(std_avg * std_multiplier, decimal_places),
                    "n_runs": len(values),
                }
            )

        if table_data:
            df = pl.DataFrame(table_data)
            total_time_pivot = df.pivot(
                values="total_time_mean",
                index=["dataset", "in", "out"],
                on="strategy",
            ).sort(["dataset", "in", "out"])
            total_time_std_pivot = df.pivot(
                values="total_time_std",
                index=["dataset", "in", "out"],
                on="strategy",
            ).sort(["dataset", "in", "out"])
            avg_time_pivot = df.pivot(
                values="avg_time_mean",
                index=["dataset", "in", "out"],
                on="strategy",
            ).sort(["dataset", "in", "out"])
            avg_time_std_pivot = df.pivot(
                values="avg_time_std",
                index=["dataset", "in", "out"],
                on="strategy",
            ).sort(["dataset", "in", "out"])
            model_tables[model_name] = {
                "total_time": total_time_pivot,
                "total_time_std": total_time_std_pivot,
                "avg_time": avg_time_pivot,
                "avg_time_std": avg_time_std_pivot,
                "raw_data": df,
            }

    metadata_table = pl.DataFrame(metadata_data) if metadata_data else None
    return model_tables, metadata_table


def create_efficiency_ranking_table(model_tables, decimal_places=3):
    ranking_tables = {}
    for model_name, tables in model_tables.items():
        if "total_time" not in tables or "avg_time" not in tables:
            continue
        ranking_df = create_ranking_table_from_pivot(
            main_df=tables["total_time"],
            tiebreak_df=tables["total_time_std"],
            decimal_places=decimal_places,
            sort_cols=["dataset", "in", "out"],
        )
        if ranking_df is not None:
            ranking_tables[model_name] = ranking_df
    return ranking_tables


def create_avg_time_ranking_table(model_tables, decimal_places=3):
    ranking_tables = {}
    for model_name, tables in model_tables.items():
        if "avg_time" not in tables:
            continue
        ranking_df = create_ranking_table_from_pivot(
            main_df=tables["avg_time"],
            tiebreak_df=tables["avg_time_std"],
            decimal_places=decimal_places,
            sort_cols=["dataset", "in", "out"],
        )
        if ranking_df is not None:
            ranking_tables[model_name] = ranking_df
    return ranking_tables


def display_efficiency_tables(
    model_tables,
    ranking_tables=None,
    avg_time_ranking_tables=None,
    metadata_table=None,
    show_metadata=True,
    decimal_places=3,
):
    for model_name, tables in model_tables.items():
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name.upper()} - TOTAL TIME (sum of time_per_iter)")
        print(f"{'='*80}")
        # Combine mean and std for display, multiply std by std_multiplier before rounding
        total_time = tables["total_time"]
        total_time_std = tables["total_time_std"]
        total_time_display = total_time.clone()
        for col in total_time.columns:
            if col in ["dataset", "in", "out"]:
                continue
            if col in total_time_std.columns:
                # Multiply std by std_multiplier before rounding and displaying
                std_col = (total_time_std[col]).round(decimal_places)
                mean_col = total_time[col].round(decimal_places)
                total_time_display = total_time_display.with_columns(
                    [(mean_col.cast(str) + "±" + std_col.cast(str)).alias(col)]
                )
            else:
                total_time_display = total_time_display.with_columns(
                    [pl.col(col).cast(str)]
                )
        print(total_time_display)
        # Show ranking table immediately after total time table
        if ranking_tables and model_name in ranking_tables:
            print(f"\n{'-'*80}")
            print(f"RANKING TABLE (by total time) FOR {model_name.upper()}")
            print(f"{'-'*80}")
            print("Strategies ranked by total time (1=fastest, lower is better)")
            print("Ties broken by standard deviation (lower std is better)")
            print("Last row shows average rank across all configurations")
            print("Last column shows which strategy is fastest most often")
            print("-" * 80)
            print(ranking_tables[model_name])
            print()
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name.upper()} - AVG TIME PER ITERATION")
        print(f"{'='*80}")
        avg_time = tables["avg_time"]
        avg_time_std = tables["avg_time_std"]
        avg_time_display = avg_time.clone()
        for col in avg_time.columns:
            if col in ["dataset", "in", "out"]:
                continue
            if col in avg_time_std.columns:
                std_col = (avg_time_std[col]).round(decimal_places)
                mean_col = avg_time[col].round(decimal_places)
                avg_time_display = avg_time_display.with_columns(
                    [(mean_col.cast(str) + "±" + std_col.cast(str)).alias(col)]
                )
            else:
                avg_time_display = avg_time_display.with_columns(
                    [pl.col(col).cast(str)]
                )
        print(avg_time_display)
        # Show ranking table immediately after avg time table
        if avg_time_ranking_tables and model_name in avg_time_ranking_tables:
            print(f"\n{'-'*80}")
            print(f"RANKING TABLE (by avg time) FOR {model_name.upper()}")
            print(f"{'-'*80}")
            print(
                "Strategies ranked by average time per iteration (1=fastest, lower is better)"
            )
            print("Ties broken by standard deviation (lower std is better)")
            print("Last row shows average rank across all configurations")
            print("Last column shows which strategy is fastest most often")
            print("-" * 80)
            print(avg_time_ranking_tables[model_name])
            print()
    if metadata_table is not None and show_metadata:
        print(f"\n{'='*80}")
        print("METADATA TABLE (All Runs and Experiments)")
        print(f"{'='*80}")
        print(metadata_table)


def save_efficiency_tables(
    model_tables,
    ranking_tables=None,
    metadata_table=None,
    output_dir="analysis/tables",
    decimal_places=3,
):
    os.makedirs(output_dir, exist_ok=True)
    suffix = ""
    if decimal_places != 3:
        suffix += f"_dec{decimal_places}"
    for model_name, tables in model_tables.items():
        total_time_path = os.path.join(
            output_dir, f"{model_name}_total_time{suffix}.csv"
        )
        avg_time_path = os.path.join(output_dir, f"{model_name}_avg_time{suffix}.csv")
        tables["total_time"].write_csv(total_time_path)
        tables["avg_time"].write_csv(avg_time_path)
        tables["raw_data"].write_csv(
            os.path.join(output_dir, f"{model_name}_efficiency_raw{suffix}.csv")
        )
        print(f"Saved total/avg time tables for {model_name} to {output_dir}")
    if ranking_tables:
        for model_name, ranking_df in ranking_tables.items():
            ranking_path = os.path.join(
                output_dir, f"{model_name}_efficiency_ranking{suffix}.csv"
            )
            ranking_df.write_csv(ranking_path)
            print(f"Saved ranking table for {model_name} to {ranking_path}")
    if metadata_table is not None:
        metadata_path = os.path.join(output_dir, f"efficiency_metadata{suffix}.csv")
        metadata_table.write_csv(metadata_path)
        print(f"Saved metadata table to {metadata_path}")


def main():
    global args
    args = parse_args(
        default_table_type="model-specific"
    )  # or "comparison" if you want
    if not args.quiet:
        print(f"Loading experiments from: {args.runs_dir}")
        print(f"Results will be displayed with {args.decimal_places} decimal places")
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
    model_tables, metadata_table = create_efficiency_tables(
        experiment_paths,
        runs_dir=args.runs_dir,
        decimal_places=args.decimal_places,
        std_multiplier=args.std_multiplier,
    )
    ranking_tables = create_efficiency_ranking_table(
        model_tables=model_tables,
        decimal_places=args.decimal_places,
    )
    avg_time_ranking_tables = create_avg_time_ranking_table(
        model_tables=model_tables,
        decimal_places=args.decimal_places,
    )
    if not args.no_display:
        display_efficiency_tables(
            model_tables,
            ranking_tables,
            avg_time_ranking_tables,
            metadata_table if args.show_metadata else None,
            show_metadata=args.show_metadata,
        )

    save_efficiency_tables(
        model_tables,
        ranking_tables,
        metadata_table,
        output_dir=args.output_dir,
        decimal_places=args.decimal_places,
    )
    if not args.quiet:
        print(f"\nTables saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
