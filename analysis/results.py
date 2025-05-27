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
    path = "runs"
    data = []
    models = []
    for run in os.listdir(path):
        p = os.path.join(path, run, "results.csv")
        if not os.path.exists(p):
            continue
        datum = {}

        with open(os.path.join("runs", run, "config.json"), "r") as f:
            j = json.load(f)
            datum["path_info"] = j["path_info"]
            datum["dataset"] = j["dataset"]
            datum["input_len"] = j["input_len"]
            datum["output_len"] = j["output_len"]
            datum["strategy"] = j["strategy"]
            datum["run"] = run
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
        # d = pl.from_dicts(d).sort(by=["dataset", "input_len", "output_len", "strategy"])
        d = pl.from_dicts(d).sort(by=["run"])

        # Pivot the data to have strategy columns with their corresponding loss values
        table_loss = d.pivot(
            index=["run"], on="strategy", values="loss"
        )

        # Save the normal loss table
        table_loss.write_csv(os.path.join("analysis", f"{model}_results_loss.csv"))
        print(table_loss)
        print("loss = mean±std(x10e-4)")
