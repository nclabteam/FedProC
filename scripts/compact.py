import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.compact import compact_experiment_runs


def clean_experiments(base_dir: Path) -> tuple[int, int]:
    """Remove experiment directories without results.csv. Returns (deleted, kept)."""
    deleted = 0
    kept = 0
    for entry in sorted(base_dir.iterdir()):
        if not entry.is_dir():
            continue
        # look for any file named results.csv anywhere under the subdir
        has_results = any(entry.rglob("results.csv"))
        if has_results:
            kept += 1
            print(f"  KEEP: {entry.name} (contains results.csv)")
        else:
            try:
                shutil.rmtree(entry)
                deleted += 1
                print(f"  DELETED: {entry.name}")
            except Exception as e:
                print(f"  ERROR deleting {entry}: {e}")
    return deleted, kept


def main():
    parser = argparse.ArgumentParser(
        description="Compact experiment runs by merging seed runs into CSV files"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="runs",
        help="Directory containing experiment subdirectories",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for compacted experiments (default: input_dir_compact)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean incomplete runs (without results.csv) in output directory before compacting",
    )
    args = parser.parse_args()

    base = Path(args.dir).expanduser().resolve()
    if not base.exists():
        print(f"Directory not found: {base}")
        sys.exit(1)
    if not base.is_dir():
        print(f"Not a directory: {base}")
        sys.exit(1)

    # Step 1: Determine output directory
    if args.output:
        output_base = Path(args.output).expanduser().resolve()
    else:
        output_base = Path(f"{base}_compact").expanduser().resolve()

    output_base.mkdir(parents=True, exist_ok=True)

    # Step 2: Find all experiment directories (those with config.json) in INPUT
    experiment_dirs = sorted(
        d for d in base.iterdir()
        if d.is_dir() and (d / "config.json").exists()
    )

    if not experiment_dirs:
        print(f"No experiments found in {base}")
        sys.exit(0)

    print(f"Found {len(experiment_dirs)} experiments\n")

    # Step 3: Copy all experiments to output directory
    print("Step 1: Copying experiments to output directory...")
    for exp_dir in experiment_dirs:
        exp_name = exp_dir.name
        output_exp_dir = output_base / exp_name
        if output_exp_dir.exists():
            shutil.rmtree(output_exp_dir)
        shutil.copytree(exp_dir, output_exp_dir)
    print(f"  Copied {len(experiment_dirs)} experiments\n")

    # Step 4: Clean output directory if requested
    if args.clean:
        print("Step 2: Cleaning incomplete runs in output directory...\n")
        deleted, kept = clean_experiments(output_base)
        print(f"  Deleted: {deleted}, Kept: {kept}\n")
        
        # Re-scan for valid experiments after cleaning
        experiment_dirs = sorted(
            d for d in output_base.iterdir()
            if d.is_dir() and (d / "config.json").exists()
        )
        print(f"Remaining experiments after clean: {len(experiment_dirs)}\n")

    # Step 5: Compact each experiment
    print("Step 3: Compacting experiments...\n")
    compacted = 0
    failed = 0

    for exp_dir in experiment_dirs:
        exp_name = exp_dir.name
        print(f"Compacting: {exp_name}")

        try:
            # Compact in the output directory
            result = compact_experiment_runs(exp_dir)
            print(f"  ✓ Runs: {result['runs']}")
            print(f"  ✓ Server rows: {result['server_rows']}")
            print(f"  ✓ Client rows: {result['client_rows']}")
            print(f"  ✓ Generated: {result['generated_files']}")
            print(f"  ✓ Deleted: {result['deleted_paths']}")
            compacted += 1

        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed += 1

    print(f"\nDone. Compacted: {compacted}, Failed: {failed}")
    print(f"Output directory: {output_base}")


if __name__ == "__main__":
    main()
