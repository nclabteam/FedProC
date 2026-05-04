#!/usr/bin/env python
import json
import logging
import multiprocessing
import os
import shutil
import sys
import time

from analysis.single import ExperimentAnalysis
from utils import Options, SetSeed
from utils.cleanup import cleanup_interrupted_run
from utils.compact import compact_experiment_runs

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
            "granularity_unit": dataset.info[0]["granularity_unit"],
            "num_clients": len(dataset.info),
        }
    )
    options.display()
    options.save()
    args = options.args

    # Copy dataset info
    shutil.copyfile(
        os.path.join(dataset.path_info),
        os.path.join(args.save_path, "info.json"),
    )

    timings = []

    try:
        for t in range(args.prev, args.times):
            print(f"\n============= Running time: {t}th =============")
            SetSeed(seed=args.seed + t).set()
            start = time.time()
            print("Creating server and clients ...")
            server = getattr(__import__("strategies"), args.strategy)(args, t)
            server.train()
            elapsed = time.time() - start
            timings.append({"run": t, "seconds": round(elapsed, 2)})
            print(f"Run {t} finished in {elapsed:.2f}s")

        timing_path = os.path.join(args.save_path, "timing.json")
        with open(timing_path, "w") as f:
            json.dump(timings, f, indent=2)
        print(f"Timings saved to {timing_path}")

        # Analyze before compact so run-level CSVs are still available
        results_path = ExperimentAnalysis(args.save_path).save()
        print(f"Analysis saved to {results_path}")

        if args.compact:
            compact_summary = compact_experiment_runs(args.save_path)
            print("Compact summary:", compact_summary)
    except KeyboardInterrupt:
        if not args.keep_useless_run:
            logging.shutdown()
            removed_path = cleanup_interrupted_run(
                save_path=args.save_path,
                project_root=args.project,
            )
            print(f"KeyboardInterrupt => This run has been removed: {removed_path}")
