import argparse
import itertools
import subprocess
import sys
from collections.abc import Sequence

ParamValue = str | float | int

BASE_ARGS = [
    "--epochs=1",
    "--times=1",
    "--scaler=Standard",
    "--project=runs_tuning",
    "--patience=20",
    "--output_len=720",
]

HYPERPARAMS = {
    "strategy": [
        "FedAvg",
        "FedAWA",
        "FedTSC",
        "FML",
        "LocalOnly",
    ],
    "join_ratio": ["0.1", "1"],
    "learning_rate": [0.0001, 0.0005, 0.001, 0.005, 0.01],
    "iterations": [100, 200, 300, 400, 500],
    "model": [
        "Linear",
        "DLinear",
        "TSMixer",
        "LinearIC",
        "SCINet",
    ],
}


def generate_combinations(
    param_dict: dict[str, list[ParamValue]],
) -> tuple[list[str], list[tuple[ParamValue, ...]]]:
    """Return hyperparameter names and their Cartesian product."""
    keys = list(param_dict.keys())
    values = [param_dict[key] for key in keys]
    return keys, list(itertools.product(*values))


def make_run_name(
    param_keys: Sequence[str],
    param_values: Sequence[ParamValue],
) -> str:
    """Build a run name from one hyperparameter combination."""
    return "_".join(
        f"{key}{value}" for key, value in zip(param_keys, param_values, strict=False)
    )


def build_command(
    param_keys: Sequence[str],
    param_values: Sequence[ParamValue],
) -> tuple[list[str], str]:
    """Build one training command and its run name."""
    combo_args = [
        f"--{key}={value}" for key, value in zip(param_keys, param_values, strict=False)
    ]
    run_name = make_run_name(param_keys=param_keys, param_values=param_values)
    command = [sys.executable, "main.py", *BASE_ARGS, *combo_args, f"--name={run_name}"]
    return command, run_name


def parse_cli_args() -> argparse.Namespace:
    """Parse hyperparameter-tuning CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without launching training runs.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Limit the number of generated runs for smoke validation.",
    )
    return parser.parse_args()


def run_all(dry_run: bool = False, max_runs: int | None = None) -> None:
    """Run or print every selected hyperparameter combination."""
    param_keys, all_combos = generate_combinations(param_dict=HYPERPARAMS)
    if max_runs is not None:
        all_combos = all_combos[:max_runs]

    for run_id, param_values in enumerate(all_combos, start=1):
        full_cmd, run_name = build_command(
            param_keys=param_keys,
            param_values=param_values,
        )

        print("==============================================")
        print(f"Run {run_id}/{len(all_combos)}: {run_name}")
        print(f"Command: {' '.join(full_cmd)}")
        print("==============================================")

        if dry_run:
            print()
            continue

        try:
            subprocess.run(full_cmd, check=True)
        except subprocess.CalledProcessError as error:
            print("\nERROR: Command failed")
            print(f"Command: {' '.join(full_cmd)}")
            print(f"Return Code: {error.returncode}")
            sys.exit(error.returncode)

        print()


if __name__ == "__main__":
    args = parse_cli_args()
    run_all(dry_run=args.dry_run, max_runs=args.max_runs)
