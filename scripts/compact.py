import argparse
import shutil
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.compact import compact_experiment_runs

COMPACT_OUTPUT_FILES = ("server.csv", "client.csv")


def has_seed_runs(experiment_dir: Path) -> bool:
    """Return whether an experiment still contains numeric seed directories."""
    return any(
        child.is_dir() and child.name.isdigit() for child in experiment_dir.iterdir()
    )


def is_compacted_experiment(experiment_dir: Path) -> bool:
    """Identify a completed compaction without treating mixed reruns as compact."""
    if not experiment_dir.is_dir():
        return False
    if not (experiment_dir / "config.json").is_file():
        return False
    if has_seed_runs(experiment_dir):
        return False
    return any(
        (experiment_dir / filename).is_file() for filename in COMPACT_OUTPUT_FILES
    )


def plan_experiments(
    experiment_dirs: Sequence[Path],
    output_base: Path,
    *,
    skip_compacted: bool,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Partition sources into copy-only, compact, and skipped experiments.

    In resume mode, an already compacted destination is trusted and skipped.
    A compacted source without a compacted destination is copied as-is, while
    only sources that still need processing enter the destructive compactor.
    """
    if not skip_compacted:
        return [], list(experiment_dirs), []

    copy_only: list[Path] = []
    to_compact: list[Path] = []
    skipped: list[Path] = []

    for experiment_dir in experiment_dirs:
        output_dir = output_base / experiment_dir.name
        if is_compacted_experiment(output_dir):
            skipped.append(experiment_dir)
        elif is_compacted_experiment(experiment_dir):
            copy_only.append(experiment_dir)
        else:
            to_compact.append(experiment_dir)

    return copy_only, to_compact, skipped


def copy_experiments(
    experiment_dirs: Sequence[Path],
    output_base: Path,
) -> list[Path]:
    """Copy experiments and return their output paths."""
    output_dirs: list[Path] = []
    for experiment_dir in experiment_dirs:
        output_dir = output_base / experiment_dir.name
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(experiment_dir, output_dir)
        output_dirs.append(output_dir)
    return output_dirs


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
    parser.add_argument(
        "--skip-compacted",
        action="store_true",
        help=(
            "Resume without rebuilding compacted experiments: trust compacted "
            "destinations and copy compacted sources without reprocessing them"
        ),
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

    if output_base == base:
        print("Output directory must differ from input directory")
        sys.exit(1)

    output_base.mkdir(parents=True, exist_ok=True)

    # Step 2: Find all experiment directories (those with config.json) in INPUT
    experiment_dirs = sorted(
        d for d in base.iterdir() if d.is_dir() and (d / "config.json").exists()
    )

    if not experiment_dirs:
        print(f"No experiments found in {base}")
        sys.exit(0)

    print(f"Found {len(experiment_dirs)} experiments\n")

    copy_only, to_compact, skipped = plan_experiments(
        experiment_dirs,
        output_base,
        skip_compacted=args.skip_compacted,
    )
    if args.skip_compacted:
        print(
            "Resume plan: "
            f"{len(skipped)} already compacted, "
            f"{len(copy_only)} copy-only, "
            f"{len(to_compact)} to compact\n"
        )

    # Step 3: Copy all experiments to output directory
    print("Step 1: Copying experiments to output directory...")
    copy_sources = [*copy_only, *to_compact]
    copy_experiments(copy_sources, output_base)
    print(f"  Copied {len(copy_sources)} experiments\n")

    # Only copied outputs may enter the destructive compactor. Sources that
    # were already compacted are copied as-is and intentionally excluded.
    experiment_dirs = [output_base / path.name for path in to_compact]

    # Step 4: Clean output directory if requested
    if args.clean:
        print("Step 2: Cleaning incomplete runs in output directory...\n")
        deleted, kept = clean_experiments(output_base)
        print(f"  Deleted: {deleted}, Kept: {kept}\n")

        experiment_dirs = [
            path
            for path in experiment_dirs
            if path.is_dir() and (path / "config.json").is_file()
        ]
        print(
            "Remaining scheduled experiments after clean: " f"{len(experiment_dirs)}\n"
        )

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
