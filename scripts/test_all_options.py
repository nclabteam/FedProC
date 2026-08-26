"""
Validate all registry entries (strategies, models, optimizers, schedulers)
have correct class-level optional/compulsory/args_update attributes.

Usage:
    python scripts/test_all_options.py              # validate attributes only
    python scripts/test_all_options.py --dry-run     # also print main.py commands
    python scripts/test_all_options.py --run         # actually run main.py for each combo
    python scripts/test_all_options.py --run --max-runs 5  # limit to 5 runs
"""

import argparse
import importlib
import itertools
import os
import subprocess
import sys
from collections.abc import Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.options import Options


def validate_registries() -> list[str]:
    """Validate all registry entries have correct class-level attributes."""
    errors = []

    for module_name in Options.COMPONENTS.values():
        module = importlib.import_module(module_name)
        components = getattr(module, module_name.upper())
        for name in components:
            for registry_name in ("optional", "compulsory"):
                try:
                    value = getattr(module, registry_name)[name]
                    if not isinstance(value, dict):
                        errors.append(
                            f"{module_name}/{name}: {registry_name} is "
                            f"{type(value).__name__}, expected dict"
                        )
                except Exception as error:
                    errors.append(
                        f"{module_name}/{name}: {registry_name} lookup failed: "
                        f"{error}"
                    )
            try:
                update = module.args_update_functions[name]
                if update is not None and not callable(update):
                    errors.append(
                        f"{module_name}/{name}: args_update is "
                        f"{type(update).__name__}, expected callable or None"
                    )
            except Exception as error:
                errors.append(
                    f"{module_name}/{name}: args_update lookup failed: {error}"
                )

    return errors


def validate_args_update_callable() -> list[str]:
    """Test that args_update functions work with argparse."""
    errors = []
    for module_name in Options.COMPONENTS.values():
        module = importlib.import_module(module_name)
        funcs = module.args_update_functions
        for name in funcs:
            try:
                func = funcs[name]
            except (AttributeError, KeyError):
                continue
            if func is None:
                continue
            try:
                p = argparse.ArgumentParser()
                func(parser=p)
            except Exception as error:
                errors.append(f"{module_name}/{name}: args_update failed: {error}")
    return errors


def build_main_commands(
    strategy_filter: Sequence[str] | None = None,
    model_filter: Sequence[str] | None = None,
    max_runs: int | None = None,
) -> list[tuple[str, str, list[str]]]:
    """Generate main.py commands for all strategy+model combos."""
    from models import MODELS
    from strategies import STRATEGIES

    # Filter out non-strategy entries
    skip_strategies = {"base", "Centralized"}
    strategies = [s for s in STRATEGIES if s not in skip_strategies]
    if strategy_filter:
        strategies = [s for s in strategies if s in strategy_filter]

    # Filter to common/simple models for testing
    test_models = ["DLinear", "Linear", "LSTM"]
    if model_filter:
        test_models = [m for m in MODELS if m in model_filter]
    models = test_models

    base_args = [
        "--dataset=ETTDataset",
        "--data_path=ETTh1.csv",
        "--input_len=96",
        "--output_len=96",
        "--device_id=0",
        "--times=1",
        "--epochs=1",
        "--iterations=1",
        "--batch_size=16",
        "--learning_rate=0.001",
        "--loss=MSE",
        "--skip_eval_train",
        "--compact",
    ]

    commands = []
    for strategy, model in itertools.product(strategies, models):
        cmd = [
            sys.executable,
            "main.py",
            *base_args,
            f"--strategy={strategy}",
            f"--model={model}",
        ]
        commands.append((strategy, model, cmd))

    if max_runs:
        commands = commands[:max_runs]

    return commands


def main() -> int:
    """Validate registries and optionally execute generated commands."""
    parser = argparse.ArgumentParser(description="Validate all registry options")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print main.py commands without running"
    )
    parser.add_argument(
        "--run", action="store_true", help="Actually run main.py for each combo"
    )
    parser.add_argument(
        "--max-runs", type=int, default=None, help="Limit number of runs"
    )
    parser.add_argument(
        "--strategy", nargs="+", default=None, help="Filter to specific strategies"
    )
    parser.add_argument(
        "--model", nargs="+", default=None, help="Filter to specific models"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Step 1: Validate registry attributes")
    print("=" * 60)
    errors = validate_registries()
    if errors:
        print(f"\nFAILED - {len(errors)} errors:")
        for e in errors:
            print(f"  {e}")
        return 1
    else:
        print("PASSED - All registry entries have valid attributes")

    print()
    print("=" * 60)
    print("Step 2: Validate args_update functions are callable")
    print("=" * 60)
    errors = validate_args_update_callable()
    if errors:
        print(f"\nFAILED - {len(errors)} errors:")
        for e in errors:
            print(f"  {e}")
        return 1
    else:
        print("PASSED - All args_update functions are callable")

    if not (args.dry_run or args.run):
        print()
        print(
            "All validations passed. Use --dry-run or --run to test main.py commands."
        )
        return 0

    print()
    print("=" * 60)
    print("Step 3: main.py command generation")
    print("=" * 60)
    commands = build_main_commands(
        strategy_filter=args.strategy,
        model_filter=args.model,
        max_runs=args.max_runs,
    )
    print(f"Generated {len(commands)} commands")

    if args.dry_run:
        for strategy, model, cmd in commands:
            print(f"  {strategy} + {model}: {' '.join(cmd)}")
        return 0

    if args.run:
        print()
        print("=" * 60)
        print("Step 4: Running main.py commands")
        print("=" * 60)
        failed = []
        for i, (strategy, model, cmd) in enumerate(commands, 1):
            print(f"[{i}/{len(commands)}] {strategy} + {model}")
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=300)
                print("  PASSED")
            except subprocess.CalledProcessError as e:
                print(f"  FAILED (exit code {e.returncode})")
                failed.append((strategy, model, e.returncode))
            except subprocess.TimeoutExpired:
                print("  TIMEOUT")
                failed.append((strategy, model, "timeout"))
            except Exception as e:
                print(f"  ERROR: {e}")
                failed.append((strategy, model, str(e)))

        print()
        if failed:
            print(f"FAILED: {len(failed)}/{len(commands)} runs")
            for s, m, err in failed:
                print(f"  {s} + {m}: {err}")
            return 1
        else:
            print(f"ALL PASSED: {len(commands)}/{len(commands)} runs")
            return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
