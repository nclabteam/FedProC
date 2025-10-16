import argparse
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Clean run subdirectories that have no results.csv"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="runs",
        help="Directory containing run subdirectories",
    )
    args = parser.parse_args()

    base = Path(args.dir).expanduser().resolve()
    if not base.exists():
        print(f"Directory not found: {base}")
        sys.exit(1)
    if not base.is_dir():
        print(f"Not a directory: {base}")
        sys.exit(1)

    deleted = 0
    kept = 0
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        # look for any file named results.csv anywhere under the subdir
        has_results = any(entry.rglob("results.csv"))
        if has_results:
            kept += 1
            print(f"KEEP: {entry} (contains results.csv)")
        else:
            try:
                shutil.rmtree(entry)
                deleted += 1
                print(f"DELETED: {entry}")
            except Exception as e:
                print(f"ERROR deleting {entry}: {e}")

    print(f"\nDone. Deleted: {deleted}, Kept: {kept}")


if __name__ == "__main__":
    main()
