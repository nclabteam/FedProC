import json
import os
import sys
from collections import defaultdict

import numpy as np
import polars as pl
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from losses import evaluation_result
from strategies.base import SharedMethods

load_data = SharedMethods.load_data


dfs = []
for dir in os.listdir("runs"):
    if not os.path.exists(os.path.join("runs", dir, "results.csv")):
        continue
    info = json.load(open(os.path.join("runs", dir, "config.json"), "r"))
    data_info = json.load(open(os.path.join(info["path_info"]), "r"))
    data = []
    for trial in os.listdir(os.path.join("runs", dir)):
        path = os.path.join("runs", dir, trial)
        if not os.path.isdir(path):
            continue
        path = os.path.join(path, "models")

        for weights in os.listdir(path):
            if not weights.endswith(".pt") and "server" not in weights:
                continue
            weights_path = os.path.join(path, weights)

            print(f"Loading {weights_path}")
            model = torch.load(weights_path, weights_only=False).cuda()
            datum = []
            for client in data_info:
                stats = client["stats"]["train"]
                scaler = getattr(__import__("scalers"), info["scaler"])(stats)
                testloader = load_data(
                    file=client["paths"]["test"],
                    sample_ratio=1.0,
                    shuffle=False,
                    scaler=scaler,
                    batch_size=info["batch_size"],
                )

                # no denorm
                subdatum = {}
                subdatum["path"] = weights_path
                subdatum["denorm"] = False
                losses = []
                for x, y in testloader:
                    x = x.cuda()
                    y = y.cuda()
                    pred = model(x)
                    loss = evaluation_result(pred, y)
                    losses.append(loss)
                for k in losses[0]:
                    subdatum[k] = np.mean([loss[k] for loss in losses])
                datum.append(subdatum)

                # denorm
                subdatum = {}
                subdatum["path"] = weights_path
                subdatum["denorm"] = True
                losses = []
                for x, y in testloader:
                    x = x.cuda()
                    y = y.cuda()
                    pred = model(x)
                    pred = torch.tensor(
                        scaler.inverse_transform(pred.cpu().detach().numpy())
                    )
                    y = torch.tensor(scaler.inverse_transform(y.cpu().detach().numpy()))
                    loss = evaluation_result(pred, y)
                    losses.append(loss)
                for k in losses[0]:
                    subdatum[k] = np.mean([loss[k] for loss in losses])
                datum.append(subdatum)

            for d in [False, True]:
                group = [r for r in datum if r["denorm"] == d]
                keys = [k for k in group[0] if isinstance(group[0][k], (int, float))]
                avg = {k: np.mean([r[k] for r in group]) for k in keys}
                avg["denorm"] = d
                model_type = "best" if "server_best" in weights else "last"
                data.append({"trial": trial, "type": model_type, **avg})
    grouped = defaultdict(list)
    for r in data:
        key = (r["denorm"], r["type"])
        grouped[key].append(r)

    summary = []
    for (denorm, t), group in grouped.items():
        keys = [k for k in group[0] if isinstance(group[0][k], (int, float))]
        avg = {
            f"{k}_avg": np.mean([r[k] for r in group]) for k in keys if k != "denorm"
        }
        std = {f"{k}_std": np.std([r[k] for r in group]) for k in keys if k != "denorm"}
        result = {
            "case": dir,
            "denorm": denorm,
            "type": t,
            "dataset": info["dataset"],
            "input_len": info["input_len"],
            "output_len": info["output_len"],
            **avg,
            **std,
        }
        summary.append(result)
    # Convert the summary into a Polars DataFrame
    df = pl.DataFrame(summary)
    print(df)
    # Save it to a CSV file
    dfs.append(df)

dfs = pl.concat(dfs)
dfs.write_csv(os.path.join("analysis", "inference.csv"))
