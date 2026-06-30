import subprocess
import sys

strategies = [
    "FedAvg",
    "FedAvgM",
    "FedAdam",
    "FedYogi",
    "FedMedian",
    "FedTrimmedAvg",
    "FedCross",
    "FedRCL",
    "FedTrend",
    "Elastic",
    "Krum",
    "MOON",
    "Centralized",
    "FedAWA",
    "FedProx",
    "FedALA",
    "FML",
    "MOTAR",
    "LocalOnly",
    "FedIT",
    "DFedSAM",
    "DFedHPO",
    "DFedProx",
    "DFedAWA",
    "DFedAvg",
    "FedSPD",
    "HydroDFL",
    "Nahar",
    "KairosDFL",
    "TidalLR",
    "ProfileNaharCoverage",
    "FedEBER",
    "FedSA_LoRA",
    "FlexLoRA",
    "LoRA_FAIR",
    "FedA2L",
    "FedDyn",
    "FedCAC",
    "FedRevIN",
    "FedDF",
    "FedMD",
    "FFA_LoRA",
]

results = {}
for s in strategies:
    print(f"Testing {s} (10 iters)...", flush=True)
    r = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--dataset",
            "ETDatasetHour",
            "--strategy",
            s,
            "--model",
            "DLinear",
            "--iterations",
            "10",
            "--times",
            "1",
            "--seed",
            "42",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if r.returncode == 0:
        results[s] = "PASS"
        print(f"  PASS")
    else:
        err = [l for l in r.stderr.split("\n") if "Error" in l or "Exception" in l]
        results[s] = err[-1] if err else f"exit {r.returncode}"
        print(f"  FAIL: {results[s]}")

print("\n===== SUMMARY =====")
passed = [s for s, v in results.items() if v == "PASS"]
failed = [s for s, v in results.items() if v != "PASS"]
print(f"PASS: {len(passed)}/{len(results)}")
if failed:
    for s in sorted(failed):
        print(f"  FAIL {s}: {results[s]}")
