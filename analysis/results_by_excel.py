import json
import os
import sys

import polars as pl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

pl.Config.set_tbl_cols(100)
pl.Config.set_tbl_rows(100)
r = 4


def extract_loss_number(s):
    """Helper: extract the number from 'mean±std' string, get the mean part as float."""
    if s is None:
        return None
    try:
        mean, _ = s.split("±")
        return float(mean)
    except:
        return None


if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else "default"
    script_path = os.path.join("scripts", f"{filename}.xlsx")
    script_df = pl.read_excel(script_path)
    data = []
    models = []
    for run in script_df["--name="].to_list():
        p = os.path.join("runs", run, "results.csv")
        if not os.path.exists(p):
            script_value = script_df.filter(pl.col("--name=") == run)[
                "script"
            ].to_list()[0]
            print(script_value)
            continue
        datum = {}

        with open(os.path.join("runs", run, "config.json"), "r") as f:
            j = json.load(f)
            datum["path_info"] = j["path_info"]
            datum["dataset"] = j["dataset"]
            datum["input_len"] = j["input_len"]
            datum["output_len"] = j["output_len"]
            datum["strategy"] = j["strategy"]
            for keyword in [
                "trend",
                "seasonal",
                "Arithmetic",
                "AutoCyclic",
                "HyperbolicLR",
                "ExpHyperbolicLR",
            ]:
                if keyword in run:
                    datum["strategy"] += f"_{keyword}"
            datum["save_local_model"] = j["save_local_model"]
            datum["model"] = j["model"]
            models.append(j["model"])
            nc = j["num_clients"]

        df = pl.read_csv(p)

        if datum["save_local_model"]:
            loss = df.filter(df["metric"].str.contains("personal_avg_test_loss"))
        else:
            loss = df.filter(df["metric"].str.contains("global_avg_test_loss"))
        datum[
            "loss"
        ] = f"{round(loss['avg_min'].to_list()[0], r):.4f}±{round(loss['std_min'].to_list()[0]*10_000, r):.4f}"
        try:
            send_to_clients_mb = max(
                df.filter(df["metric"].str.contains("send_mb"))["avg_min"].to_list()[0],
                0,
            )
        except:
            send_to_clients_mb = -999999

        df = pl.read_csv(os.path.join("runs", run, "0", "results", "client_000.csv"))
        send_to_server = max(0, df["send_mb"].to_list()[0])
        datum["communication_mb"] = send_to_clients_mb + nc * send_to_server
        datum["num_clients"] = nc

        data.append(datum)

    for model in list(set(models)):
        print(f"{model = }")
        d = [d for d in data if d["model"] == model]
        d = pl.from_dicts(d).sort(by=["dataset", "input_len", "output_len", "strategy"])

        # Pivot the data to have strategy columns with their corresponding loss values
        table_loss = d.pivot(
            index=["dataset", "input_len", "output_len"], on="strategy", values="loss"
        )

        # Save the normal loss table
        table_loss.write_csv(os.path.join("analysis", f"{model}_results_loss.csv"))
        print(table_loss)
        print("loss = mean±std(x10e-4)")

        # Now calculate the relative increase table
        strategy_cols = [
            col
            for col in table_loss.columns
            if col not in ["dataset", "input_len", "output_len"]
        ]

        # Build a new table: percentage change
        relative_rows = []
        for row in table_loss.iter_rows(named=True):
            fedavg_val = extract_loss_number(row.get("FedAvg", None))
            relative_row = {k: row[k] for k in ["dataset", "input_len", "output_len"]}
            if fedavg_val is None or fedavg_val == 0:
                # If no FedAvg, set all % to None
                for col in strategy_cols:
                    relative_row[col] = None
            else:
                for col in strategy_cols:
                    model_val = extract_loss_number(row.get(col, None))
                    if model_val is None:
                        relative_row[col] = None
                    else:
                        relative_row[col] = round(
                            100 * (fedavg_val - model_val) / fedavg_val, 2
                        )  # in percentage
            relative_rows.append(relative_row)

        table_relative = pl.from_dicts(relative_rows)
        table_relative.write_csv(
            os.path.join("analysis", f"{model}_results_relative.csv")
        )
        print(table_relative)
        print("value = % increase compared to FedAvg")
