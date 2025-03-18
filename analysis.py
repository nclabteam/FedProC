import json
import os

import polars as pl

r = 4

if __name__ == "__main__":
    script_df = pl.read_excel("scripts.xlsx")
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
            datum["dataset"] = j["dataset"]
            datum["input_len"] = j["input_len"]
            datum["output_len"] = j["output_len"]
            datum["strategy"] = j["strategy"]
            datum["save_local_model"] = j["save_local_model"]
            datum["model"] = j["model"]
            models.append(j["model"])
            nc = j["num_clients"]

        df = pl.read_csv(p)
        print(datum)
        if datum["save_local_model"]:
            loss = df.filter(df["metric"].str.contains("personal_avg_test_loss"))
        else:
            loss = df.filter(df["metric"].str.contains("global_avg_test_loss"))

        datum[
            "loss"
        ] = f"{round(loss['avg_min'].to_list()[0], r):.4f}±{round(loss['std_min'].to_list()[0]*10_000, r):.4f}"
        send_to_clients_mb = max(
            df.filter(df["metric"].str.contains("send_mb"))["avg_min"].to_list()[0], 0
        )

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
        d = d.pivot(
            index=["dataset", "input_len", "output_len"], on="strategy", values="loss"
        )

        # Save the transformed data
        d.write_csv("results.csv")
        print(d)
        print("loss = mean±std(x10e-4)")
