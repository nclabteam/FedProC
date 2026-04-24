import json
import shutil
from pathlib import Path
from typing import Any

import polars as pl

COMPACT_DIR_NAME = "compact"
ALLOWLIST_DIRS = ("logs", "models_info")
CLIENT_GLOB = "client_*.csv"


def _seed_run_dirs(save_path: Path) -> list[Path]:
    return sorted(
        child
        for child in save_path.iterdir()
        if child.is_dir() and child.name.isdigit()
    )


def _annotate_frame(frame: pl.DataFrame, **columns: Any) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    result = frame.with_row_index("step")
    for name, value in columns.items():
        result = result.with_columns(pl.lit(value).alias(name))
    return result


def compact_experiment_runs(save_path: str | Path) -> dict[str, Any]:
    experiment_dir = Path(save_path).expanduser().resolve()
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    if not experiment_dir.is_dir():
        raise ValueError(f"Experiment path is not a directory: {experiment_dir}")

    compact_dir = experiment_dir / COMPACT_DIR_NAME
    compact_dir.mkdir(parents=True, exist_ok=True)

    server_frames: list[pl.DataFrame] = []
    client_frames: list[pl.DataFrame] = []
    delete_targets: list[Path] = []

    run_dirs = _seed_run_dirs(experiment_dir)
    for run_dir in run_dirs:
        seed = int(run_dir.name)
        results_dir = run_dir / "results"
        server_path = results_dir / "server.csv"
        if server_path.exists():
            server_frames.append(_annotate_frame(pl.read_csv(server_path), seed=seed))

        for client_path in sorted(results_dir.glob(CLIENT_GLOB)):
            client_frames.append(
                _annotate_frame(
                    pl.read_csv(client_path),
                    seed=seed,
                    client=client_path.stem,
                )
            )
            delete_targets.append(client_path)

        for dirname in ALLOWLIST_DIRS:
            target = run_dir / dirname
            if target.exists():
                delete_targets.append(target)

    generated_files: list[str] = []
    if server_frames:
        server_output = compact_dir / "server.csv"
        pl.concat(server_frames, how="diagonal_relaxed").write_csv(server_output)
        generated_files.append(str(server_output.relative_to(experiment_dir)))
    if client_frames:
        client_output = compact_dir / "clients.csv"
        pl.concat(client_frames, how="diagonal_relaxed").write_csv(client_output)
        generated_files.append(str(client_output.relative_to(experiment_dir)))

    deleted_relative_paths: list[str] = []
    for target in delete_targets:
        if not target.exists():
            continue
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=False)
        else:
            target.unlink()
        deleted_relative_paths.append(str(target.relative_to(experiment_dir)))

    summary = {
        "runs": len(run_dirs),
        "server_rows": sum(frame.height for frame in server_frames),
        "client_rows": sum(frame.height for frame in client_frames),
        "generated_files": generated_files,
        "deleted_paths": sorted(deleted_relative_paths),
    }
    with open(compact_dir / "manifest.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary
