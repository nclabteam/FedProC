import argparse
import os
import sys
from pathlib import Path

import polars as pl

pl.Config.set_tbl_cols(100)
pl.Config.set_tbl_rows(100)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.analysis import (
    _load_experiment_data,
    create_ranking_table_from_pivot,
    extract_loss_values,
    filter_experiments,
    get_experiment_config,
    get_experiment_names_from_excel,
    get_experiment_paths,
    load_all_experiments,
    parse_args,
)


# Helper: convert size units (input in MB)
def convert_size_units(value_mb, unit):
    if unit.upper() in ("MB", "M"):
        return value_mb
    if unit.upper() in ("GB", "G"):
        return value_mb / 1024.0
    raise ValueError(f"Unsupported size unit: {unit}")


def create_communication_tables(
    experiment_paths,
    runs_dir="runs",
    decimal_places=3,
    size_unit="MB",
    max_lines=None,
):
    """
    Create communication tables similar to efficiency.py.

    Behavior:
      - Metric keys: 'send_mb' and 'recv_mb' (per-run value or list). Multiply by 2 before using.
      - Convert from MB to desired unit (MB or GB) when building tables.
    """
    all_experiments = []
    for exp_path in experiment_paths:
        exp_dir = os.path.join(runs_dir, exp_path)
        if os.path.isdir(exp_dir) and os.path.exists(
            os.path.join(exp_dir, "results.csv")
        ):
            datum = _load_experiment_data(experiment_dir=exp_dir, max_lines=max_lines)
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

                # Extract send_mb and recv_mb metrics (could be single value or list)
                send_val = run.get("send_mb", None)
                recv_val = run.get("recv_mb", None)

                # fallbacks if stored differently
                if send_val is None:
                    send_val = extract_loss_values(run, "send_mb")
                if recv_val is None:
                    recv_val = extract_loss_values(run, "recv_mb")

                if (send_val is None and recv_val is None) or run_name == "avg":
                    continue

                # helper to process a value (list or scalar): multiply by 2, return (total_mb, avg_mb)
                def _process_metric(val):
                    if val is None:
                        return None, None
                    if isinstance(val, (list, tuple)):
                        scaled = [float(v) * 2.0 for v in val]
                        total_mb = float(sum(scaled))
                        num_rounds = len(scaled)
                        avg_mb = total_mb / num_rounds if num_rounds > 0 else 0.0
                        return total_mb, avg_mb
                    else:
                        try:
                            v = float(val)
                        except Exception:
                            return None, None
                        scaled_v = v * 2.0
                        # try infer rounds from other metrics (fallback to 1)
                        num_rounds = 1
                        for loss_key in [
                            "global_avg_train_loss",
                            "personal_avg_train_loss",
                            "global_avg_test_loss",
                            "personal_avg_test_loss",
                        ]:
                            loss_seq = run.get(loss_key)
                            if (
                                isinstance(loss_seq, (list, tuple))
                                and len(loss_seq) > 1
                            ):
                                num_rounds = len(loss_seq)
                                break
                        total_mb = scaled_v
                        avg_mb = total_mb / num_rounds if num_rounds > 0 else total_mb
                        return total_mb, avg_mb

                total_send_mb, avg_send_mb = _process_metric(send_val)
                total_recv_mb, avg_recv_mb = _process_metric(recv_val)

                # if both metrics are missing skip
                if total_send_mb is None and total_recv_mb is None:
                    continue

                # convert to desired unit (None-safe)
                def _conv(x):
                    return convert_size_units(x, size_unit) if x is not None else None

                total_converted_send = _conv(total_send_mb)
                avg_converted_send = _conv(avg_send_mb)
                total_converted_recv = _conv(total_recv_mb)
                avg_converted_recv = _conv(avg_recv_mb)

                combination_data.setdefault(combination_key, []).append(
                    (
                        total_converted_send,
                        avg_converted_send,
                        total_converted_recv,
                        avg_converted_recv,
                    )
                )

        table_data = []
        for combination_key, values in combination_data.items():
            strategy, dataset, input_len, output_len = combination_key
            total_send_vals = [v[0] for v in values if v[0] is not None]
            avg_send_vals = [v[1] for v in values if v[1] is not None]
            total_recv_vals = [v[2] for v in values if v[2] is not None]
            avg_recv_vals = [v[3] for v in values if v[3] is not None]

            mean_total_send = (
                pl.Series(total_send_vals).mean() if total_send_vals else None
            )
            std_total_send = (
                pl.Series(total_send_vals).std() if len(total_send_vals) > 1 else 0.0
            )
            mean_avg_send = pl.Series(avg_send_vals).mean() if avg_send_vals else None
            std_avg_send = (
                pl.Series(avg_send_vals).std() if len(avg_send_vals) > 1 else 0.0
            )

            mean_total_recv = (
                pl.Series(total_recv_vals).mean() if total_recv_vals else None
            )
            std_total_recv = (
                pl.Series(total_recv_vals).std() if len(total_recv_vals) > 1 else 0.0
            )
            mean_avg_recv = pl.Series(avg_recv_vals).mean() if avg_recv_vals else None
            std_avg_recv = (
                pl.Series(avg_recv_vals).std() if len(avg_recv_vals) > 1 else 0.0
            )

            table_data.append(
                {
                    "strategy": strategy,
                    "dataset": dataset,
                    "in": input_len,
                    "out": output_len,
                    "total_send_mean": (
                        round(mean_total_send, decimal_places)
                        if mean_total_send is not None
                        else None
                    ),
                    "total_send_std": round(std_total_send, decimal_places),
                    "avg_send_mean": (
                        round(mean_avg_send, decimal_places)
                        if mean_avg_send is not None
                        else None
                    ),
                    "avg_send_std": round(std_avg_send, decimal_places),
                    "total_recv_mean": (
                        round(mean_total_recv, decimal_places)
                        if mean_total_recv is not None
                        else None
                    ),
                    "total_recv_std": round(std_total_recv, decimal_places),
                    "avg_recv_mean": (
                        round(mean_avg_recv, decimal_places)
                        if mean_avg_recv is not None
                        else None
                    ),
                    "avg_recv_std": round(std_avg_recv, decimal_places),
                    "n_runs": len(values),
                }
            )

        if table_data:
            df = pl.DataFrame(table_data)
            total_send_pivot = df.pivot(
                values="total_send_mean",
                index=["dataset", "in", "out"],
                on="strategy",
            ).sort(["dataset", "in", "out"])
            total_send_std_pivot = df.pivot(
                values="total_send_std",
                index=["dataset", "in", "out"],
                on="strategy",
            ).sort(["dataset", "in", "out"])
            avg_send_pivot = df.pivot(
                values="avg_send_mean",
                index=["dataset", "in", "out"],
                on="strategy",
            ).sort(["dataset", "in", "out"])
            avg_send_std_pivot = df.pivot(
                values="avg_send_std",
                index=["dataset", "in", "out"],
                on="strategy",
            ).sort(["dataset", "in", "out"])

            total_recv_pivot = df.pivot(
                values="total_recv_mean",
                index=["dataset", "in", "out"],
                on="strategy",
            ).sort(["dataset", "in", "out"])
            total_recv_std_pivot = df.pivot(
                values="total_recv_std",
                index=["dataset", "in", "out"],
                on="strategy",
            ).sort(["dataset", "in", "out"])
            avg_recv_pivot = df.pivot(
                values="avg_recv_mean",
                index=["dataset", "in", "out"],
                on="strategy",
            ).sort(["dataset", "in", "out"])
            avg_recv_std_pivot = df.pivot(
                values="avg_recv_std",
                index=["dataset", "in", "out"],
                on="strategy",
            ).sort(["dataset", "in", "out"])

            model_tables[model_name] = {
                "total_send": total_send_pivot,
                "total_send_std": total_send_std_pivot,
                "avg_send": avg_send_pivot,
                "avg_send_std": avg_send_std_pivot,
                "total_recv": total_recv_pivot,
                "total_recv_std": total_recv_std_pivot,
                "avg_recv": avg_recv_pivot,
                "avg_recv_std": avg_recv_std_pivot,
                "raw_data": df,
            }

    metadata_table = pl.DataFrame(metadata_data) if metadata_data else None
    return model_tables, metadata_table


def create_communication_ranking_table(model_tables, decimal_places=3):
    ranking_tables = {}
    for model_name, tables in model_tables.items():
        if "total_send" not in tables or "avg_send" not in tables:
            continue
        ranking_df = create_ranking_table_from_pivot(
            main_df=tables["total_send"],
            tiebreak_df=tables["total_send_std"],
            decimal_places=decimal_places,
            sort_cols=["dataset", "in", "out"],
        )
        if ranking_df is not None:
            ranking_tables[model_name] = ranking_df
    return ranking_tables


def display_communication_tables(
    model_tables,
    ranking_tables=None,
    metadata_table=None,
    show_metadata=True,
    decimal_places=3,
    size_unit="MB",
):
    unit_label = size_unit.upper()
    for model_name, tables in model_tables.items():
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name.upper()} - TOTAL SENT ({unit_label})")
        print(f"{'='*80}")
        total_send = tables["total_send"]
        total_std = tables["total_send_std"]
        total_display = total_send.clone()
        for col in total_send.columns:
            if col in ["dataset", "in", "out"]:
                continue
            if col in total_std.columns:
                std_col = (total_std[col]).round(decimal_places)
                mean_col = total_send[col].round(decimal_places)
                total_display = total_display.with_columns(
                    [(mean_col.cast(str) + "±" + std_col.cast(str)).alias(col)]
                )
            else:
                total_display = total_display.with_columns([pl.col(col).cast(str)])
        print(total_display)

        if ranking_tables and model_name in ranking_tables:
            print(f"\n{'-'*80}")
            print(f"RANKING TABLE (by total sent) FOR {model_name.upper()}")
            print(f"{'-'*80}")
            print(
                "Strategies ranked by total communication (1=least) - lower is better"
            )
            print("-" * 80)
            print(ranking_tables[model_name])
            print()

        print(f"\n{'='*80}")
        print(f"MODEL: {model_name.upper()} - AVG SENT PER ROUND ({unit_label})")
        print(f"{'='*80}")
        avg_send = tables["avg_send"]
        avg_std = tables["avg_send_std"]
        avg_display = avg_send.clone()
        for col in avg_send.columns:
            if col in ["dataset", "in", "out"]:
                continue
            if col in avg_std.columns:
                std_col = (avg_std[col]).round(decimal_places)
                mean_col = avg_send[col].round(decimal_places)
                avg_display = avg_display.with_columns(
                    [(mean_col.cast(str) + "±" + std_col.cast(str)).alias(col)]
                )
            else:
                avg_display = avg_display.with_columns([pl.col(col).cast(str)])
        print(avg_display)
        print()

    if metadata_table is not None and show_metadata:
        print(f"\n{'='*80}")
        print("METADATA TABLE (All Runs and Experiments)")
        print(f"{'='*80}")
        print(metadata_table)


def save_communication_tables(
    model_tables,
    ranking_tables=None,
    metadata_table=None,
    output_dir="analysis/tables",
    decimal_places=3,
    size_unit="MB",
):
    os.makedirs(output_dir, exist_ok=True)
    suffix = ""
    if decimal_places != 3:
        suffix += f"_dec{decimal_places}"
    if size_unit.upper() != "MB":
        suffix += f"_{size_unit}"

    for model_name, tables in model_tables.items():
        total_path = os.path.join(output_dir, f"{model_name}_total_send{suffix}.csv")
        avg_path = os.path.join(output_dir, f"{model_name}_avg_send{suffix}.csv")
        tables["total_send"].write_csv(total_path)
        tables["avg_send"].write_csv(avg_path)
        tables["raw_data"].write_csv(
            os.path.join(output_dir, f"{model_name}_comm_raw{suffix}.csv")
        )
        print(f"Saved communication tables for {model_name} to {output_dir}")

    if ranking_tables:
        for model_name, ranking_df in ranking_tables.items():
            ranking_path = os.path.join(
                output_dir, f"{model_name}_comm_ranking{suffix}.csv"
            )
            ranking_df.write_csv(ranking_path)
            print(f"Saved ranking table for {model_name} to {ranking_path}")

    if metadata_table is not None:
        metadata_path = os.path.join(output_dir, f"comm_metadata{suffix}.csv")
        metadata_table.write_csv(metadata_path)
        print(f"Saved metadata table to {metadata_path}")


def main():
    global args
    args = parse_args(default_table_type="model-specific")

    # Support a lightweight custom flag --size-unit without modifying global parse_args:
    raw_size_unit = "MB"
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a.startswith("--size-unit="):
            raw_size_unit = a.split("=", 1)[1]
        elif a == "--size-unit" and i + 1 < len(argv):
            raw_size_unit = argv[i + 1]
    size_unit = raw_size_unit.upper()

    if not args.quiet:
        print(f"Loading experiments from: {args.runs_dir}")
        print(f"Results will be displayed with {args.decimal_places} decimal places")
        print(f"Size unit: {size_unit}")

    experiments = load_all_experiments(runs_dir=args.runs_dir, max_lines=args.max_lines)
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

    model_tables, metadata_table = create_communication_tables(
        experiment_paths,
        runs_dir=args.runs_dir,
        decimal_places=args.decimal_places,
        size_unit=size_unit,
        max_lines=args.max_lines,
    )

    ranking_tables = create_communication_ranking_table(
        model_tables=model_tables, decimal_places=args.decimal_places
    )

    if not args.no_display:
        display_communication_tables(
            model_tables,
            ranking_tables=ranking_tables,
            metadata_table=metadata_table if args.show_metadata else None,
            show_metadata=args.show_metadata,
            decimal_places=args.decimal_places,
            size_unit=size_unit,
        )

    save_communication_tables(
        model_tables,
        ranking_tables=ranking_tables,
        metadata_table=metadata_table,
        output_dir=args.output_dir,
        decimal_places=args.decimal_places,
        size_unit=size_unit,
    )

    if not args.quiet:
        print(f"\nCommunication tables saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
