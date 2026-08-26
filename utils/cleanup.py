"""Safely remove interrupted experiment run directories."""

import os
import shutil
import stat
from collections.abc import Callable
from pathlib import Path
from types import TracebackType


def resolve_cleanup_path(save_path: str, project_root: str) -> Path:
    """Resolve a run path and reject targets outside or equal to the project root."""
    resolved_save_path = Path(save_path).resolve()
    resolved_project_root = Path(project_root).resolve()

    try:
        is_within_project = os.path.commonpath(
            [str(resolved_save_path), str(resolved_project_root)]
        ) == str(resolved_project_root)
    except ValueError as error:
        raise ValueError(
            f"Cleanup path {resolved_save_path} is not under project root "
            f"{resolved_project_root}."
        ) from error

    if not is_within_project or resolved_save_path == resolved_project_root:
        raise ValueError(
            f"Refusing to delete {resolved_save_path}; expected a run directory under "
            f"{resolved_project_root}."
        )

    return resolved_save_path


def cleanup_interrupted_run(save_path: str, project_root: str) -> Path:
    """Remove an interrupted run, retrying read-only paths after chmod."""

    def handle_remove_error(
        func: Callable[[str], object],
        path: str,
        exc_info: tuple[type[BaseException], BaseException, TracebackType],
    ) -> None:
        _, error, _ = exc_info
        if not isinstance(error, PermissionError):
            raise error
        os.chmod(path, stat.S_IWRITE)
        func(path)

    resolved_save_path = resolve_cleanup_path(
        save_path=save_path,
        project_root=project_root,
    )
    shutil.rmtree(resolved_save_path, onerror=handle_remove_error)
    return resolved_save_path
