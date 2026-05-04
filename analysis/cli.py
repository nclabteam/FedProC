"""
Unified CLI entry point for analysis tools.

Usage:
    python -m analysis single --experiment runs/exp19
    python -m analysis multi --runs-dir runs --metric loss
    python -m analysis landscape --experiment runs/exp19
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m analysis <command> [args]")
        print()
        print("Commands:")
        print("  single      Per-experiment analysis (aggregate across runs)")
        print("  multi       Cross-experiment analysis (pivot/ranking tables)")
        print("  landscape   Loss landscape visualization")
        print()
        print("Run 'python -m analysis <command> --help' for details.")
        sys.exit(1)

    command = sys.argv[1]
    # Pass remaining args to the subcommand
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "single":
        from analysis.single import main as single_main
        single_main()
    elif command == "multi":
        from analysis.multi import main as multi_main
        multi_main()
    elif command == "landscape":
        from analysis.landscape import main as landscape_main
        landscape_main()
    else:
        print(f"Unknown command: {command}")
        print("Available: single, multi, landscape")
        sys.exit(1)


if __name__ == "__main__":
    main()
