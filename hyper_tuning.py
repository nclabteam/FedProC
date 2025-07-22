import itertools
import subprocess
import sys

# ----------- USER CONFIGURATION ------------

base_args = [
    "--iterations=2",
    "--epochs=1",
    "--times=1",
    "--model=Linear",
    "--scaler=StandardScaler",
    "--project=runs_test",
]

hyperparams = {
    "strategy": [
        "Centralized",
        "DFL",
        "Elastic",
        "FedAdam",
        "FedALA",
        "FedAvg",
        "FedAvgM",
        "FedAWA",
        "FedCAC",
        "FedDyn",
        "FedMedian",
        "FedProx",
        "FedTrend",
        "FedTrimmedAvg",
        "FedYogi",
        "FML",
        "Krum",
        "LocalOnly",
    ],
    "join_ratio": ["0.1", "1"],
    "device_id": ["0,1", "0"],
}

# ----------- SCRIPT START ------------------


def generate_combinations(param_dict):
    keys = list(param_dict.keys())
    values = [param_dict[k] for k in keys]
    product = list(itertools.product(*values))
    return keys, product


def make_run_name(param_keys, param_values):
    parts = [f"{k}{v}" for k, v in zip(param_keys, param_values)]
    return "_".join(parts)


def run_all():
    param_keys, all_combos = generate_combinations(hyperparams)

    for run_id, param_values in enumerate(all_combos):
        combo_args = [f"--{k}={v}" for k, v in zip(param_keys, param_values)]
        run_name = make_run_name(param_keys, param_values)

        print("==============================================")
        print(f"Running with params: {' '.join(combo_args)}")
        print("==============================================")

        full_cmd = (
            ["python", "main.py"] + base_args + combo_args + [f"--name={run_name}"]
        )

        try:
            subprocess.run(full_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print("\n❌ ERROR: Command failed")
            print(f"Command: {' '.join(full_cmd)}")
            print(f"Return Code: {e.returncode}")
            sys.exit(1)  # Immediately stop execution

        print()


if __name__ == "__main__":
    run_all()
