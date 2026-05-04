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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def validate_registries():
    """Validate all registry entries have correct class-level attributes."""
    errors = []

    # Strategies
    from strategies import STRATEGIES
    from strategies import args_update_functions as s_args
    from strategies import compulsory as s_comp
    from strategies import optional as s_opt

    skip = {"base"}
    for name in STRATEGIES:
        if name in skip:
            continue
        try:
            opt = s_opt[name]
            if not isinstance(opt, dict):
                errors.append(
                    f"strategies/{name}: optional is {type(opt).__name__}, expected dict"
                )
        except Exception as e:
            errors.append(f"strategies/{name}: optional lookup failed: {e}")
        try:
            comp = s_comp[name]
            if not isinstance(comp, dict):
                errors.append(
                    f"strategies/{name}: compulsory is {type(comp).__name__}, expected dict"
                )
        except Exception as e:
            errors.append(f"strategies/{name}: compulsory lookup failed: {e}")
        try:
            func = s_args[name]
            if func is not None and not callable(func):
                errors.append(
                    f"strategies/{name}: args_update is {type(func).__name__}, expected callable or None"
                )
        except Exception as e:
            errors.append(f"strategies/{name}: args_update lookup failed: {e}")

    # Models
    from models import MODELS
    from models import args_update_functions as m_args
    from models import optional as m_opt

    for name in MODELS:
        try:
            opt = m_opt[name]
            if not isinstance(opt, dict):
                errors.append(
                    f"models/{name}: optional is {type(opt).__name__}, expected dict"
                )
        except Exception as e:
            errors.append(f"models/{name}: optional lookup failed: {e}")
        try:
            func = m_args[name]
            if func is not None and not callable(func):
                errors.append(
                    f"models/{name}: args_update is {type(func).__name__}, expected callable or None"
                )
        except Exception as e:
            errors.append(f"models/{name}: args_update lookup failed: {e}")

    # Optimizers
    from optimizers import OPTIMIZERS
    from optimizers import args_update_functions as o_args
    from optimizers import optional as o_opt

    for name in OPTIMIZERS:
        try:
            opt = o_opt[name]
            if not isinstance(opt, dict):
                errors.append(
                    f"optimizers/{name}: optional is {type(opt).__name__}, expected dict"
                )
        except Exception as e:
            errors.append(f"optimizers/{name}: optional lookup failed: {e}")
        try:
            func = o_args[name]
            if func is not None and not callable(func):
                errors.append(
                    f"optimizers/{name}: args_update is {type(func).__name__}, expected callable or None"
                )
        except Exception as e:
            errors.append(f"optimizers/{name}: args_update lookup failed: {e}")

    # Schedulers
    from schedulers import SCHEDULERS
    from schedulers import args_update_functions as sc_args
    from schedulers import optional as sc_opt

    for name in SCHEDULERS:
        try:
            opt = sc_opt[name]
            if not isinstance(opt, dict):
                errors.append(
                    f"schedulers/{name}: optional is {type(opt).__name__}, expected dict"
                )
        except Exception as e:
            errors.append(f"schedulers/{name}: optional lookup failed: {e}")
        try:
            func = sc_args[name]
            if func is not None and not callable(func):
                errors.append(
                    f"schedulers/{name}: args_update is {type(func).__name__}, expected callable or None"
                )
        except Exception as e:
            errors.append(f"schedulers/{name}: args_update lookup failed: {e}")

    return errors


def validate_args_update_callable():
    """Test that args_update functions work with argparse."""
    errors = []
    skip = {"base"}
    registries = [
        ("strategies", "args_update_functions"),
        ("models", "args_update_functions"),
        ("optimizers", "args_update_functions"),
        ("schedulers", "args_update_functions"),
    ]
    for mod_name, attr_name in registries:
        mod = importlib.import_module(mod_name)
        funcs = getattr(mod, attr_name)
        for name in funcs:
            if name in skip:
                continue
            try:
                func = funcs[name]
            except (AttributeError, KeyError):
                continue
            if func is None:
                continue
            try:
                p = argparse.ArgumentParser()
                func(p)
            except Exception as e:
                errors.append(f"{mod_name}/{name}: args_update failed: {e}")
    return errors


def build_main_commands(strategy_filter=None, model_filter=None, max_runs=None):
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


def main():
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
    commands = build_main_commands(args.strategy, args.model, args.max_runs)
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
                print(f"  PASSED")
            except subprocess.CalledProcessError as e:
                print(f"  FAILED (exit code {e.returncode})")
                failed.append((strategy, model, e.returncode))
            except subprocess.TimeoutExpired:
                print(f"  TIMEOUT")
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
