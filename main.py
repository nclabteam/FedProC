#!/usr/bin/env python
import multiprocessing
import os
import sys
import time
from collections import defaultdict

import numpy as np
import polars as pl

from utils import Options, SetSeed

FILE = os.path.abspath(__file__)
ROOT = os.path.dirname(FILE)  # root directory
if ROOT not in sys.path:
    sys.path.append(ROOT)  # add ROOT to PATH
ROOT = os.path.relpath(ROOT, os.getcwd())  # relative

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    options = Options(root=ROOT).parse_options()
    options.fix_args()
    dataset = getattr(__import__("data_factory"), options.args.dataset)(
        configs=options.args
    )
    dataset.execute()
    options.update_args(
        {
            "path_info": dataset.path_info,
            "dataset_path": dataset.path_save,
            "input_channels": len(dataset.column_train),
            "output_channels": len(dataset.column_target),
            "num_clients": len(dataset.info["clients"]),
        }
    )
    options.display()
    options.save()
    args = options.args
    time_per_experiment = []
    stats = defaultdict(lambda: {"min": [], "max": []})

    try:
        for t in range(args.prev, args.times):
            print(f"\n============= Running time: {t}th =============")
            SetSeed(seed=args.seed + t).set()
            start = time.time()
            print("Creating server and clients ...")
            server = getattr(__import__("strategies"), args.strategy)(args, t)
            server.train()
            for key, value in server.metrics.items():
                stats[key]["min"].append(min(value))
                stats[key]["max"].append(max(value))
            stats["time_per_experiment"]["min"].append(time.time() - start)
            stats["time_per_experiment"]["max"].append(time.time() - start)
        rows = []
        for metric, stats in stats.items():
            row = {
                "metric": metric,
                "avg_min": np.mean(stats["min"]),
                "std_min": np.std(stats["min"]),
                "avg_max": np.mean(stats["max"]),
                "std_max": np.std(stats["max"]),
            }
            rows.append(row)
        stats = pl.DataFrame(rows)
        stats.write_csv(os.path.join(args.save_path, "results.csv"))
        print(stats)
    except KeyboardInterrupt:
        if not args.keep_useless_run:
            import logging

            logging.shutdown()
            os.system(f"rm -rf {args.save_path}")
            print(f"KeyboardInterrupt => This run has been removed.")
