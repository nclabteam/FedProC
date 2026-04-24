import shutil
from pathlib import Path
from typing import Any

import polars as pl

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
    for name, value in columns.items():
        frame = frame.with_columns(pl.lit(value).alias(name))
    return frame


def compact_experiment_runs(save_path: str | Path) -> dict[str, Any]:
    experiment_dir = Path(save_path).expanduser().resolve()
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    if not experiment_dir.is_dir():
        raise ValueError(f"Experiment path is not a directory: {experiment_dir}")

    server_frames: list[pl.DataFrame] = []
    client_frames: list[pl.DataFrame] = []

    run_dirs = _seed_run_dirs(experiment_dir)
    for run_dir in run_dirs:
        run = int(run_dir.name)
        results_dir = run_dir / "results"
        server_path = results_dir / "server.csv"
        if server_path.exists():
            server_frames.append(_annotate_frame(pl.read_csv(server_path), run=run))

        for client_path in sorted(results_dir.glob(CLIENT_GLOB)):
            client_frames.append(
                _annotate_frame(
                    pl.read_csv(client_path),
                    run=run,
                    client=client_path.stem,
                )
            )

    generated_files: list[str] = []
    if server_frames:
        server_output = experiment_dir / "server.csv"
        pl.concat(server_frames, how="diagonal_relaxed").write_csv(server_output)
        generated_files.append("server.csv")
    if client_frames:
        client_output = experiment_dir / "client.csv"
        pl.concat(client_frames, how="diagonal_relaxed").write_csv(client_output)
        generated_files.append("client.csv")

    deleted_paths: list[str] = []
    for run_dir in run_dirs:
        if run_dir.exists():
            shutil.rmtree(run_dir)
            deleted_paths.append(run_dir.name)

    info_json = experiment_dir / "info.json"
    if info_json.exists():
        info_json.unlink()
        deleted_paths.append("info.json")

    return {
        "runs": len(run_dirs),
        "server_rows": sum(f.height for f in server_frames),
        "client_rows": sum(f.height for f in client_frames),
        "generated_files": generated_files,
        "deleted_paths": sorted(deleted_paths),
    }
