import json
import os

import polars as pl

d = 4

if __name__ == "__main__":
    data = []
    for run in pl.read_excel("scripts.xlsx")["--name="].to_list():
        p = os.path.join("runs", run, "results.csv")
        if not os.path.exists(p):
            print(f"{p = }")
            continue
        datum = {}

        with open(os.path.join("runs", run, "config.json"), "r") as f:
            j = json.load(f)
            datum["dataset"] = j["dataset"]
            datum["input_len"] = j["input_len"]
            datum["output_len"] = j["output_len"]
            datum["strategy"] = j["strategy"]
            nc = j["num_clients"]

        df = pl.read_csv(p)

        if datum["strategy"] != "Centralized":
            loss = df.filter(df["metric"].str.contains("personal_avg_test_loss"))
        else:
            loss = df.filter(df["metric"].str.contains("global_avg_test_loss"))

        datum[
            "loss"
        ] = f"{round(loss['avg_min'].to_list()[0], d)}±{round(loss['std_min'].to_list()[0], d)}"
        send_to_clients_mb = max(
            df.filter(df["metric"].str.contains("send_mb"))["avg_min"].to_list()[0], 0
        )

        df = pl.read_csv(os.path.join("runs", run, "0", "results", "client_000.csv"))
        send_to_server = max(0, df["send_mb"].to_list()[0])
        datum["communication_mb"] = send_to_clients_mb + nc * send_to_server
        datum["num_clients"] = nc

        data.append(datum)

    data = pl.from_dicts(data).sort(by=["dataset", "input_len", "output_len", "loss"])
    data.write_csv("results.csv")
    print(data)
