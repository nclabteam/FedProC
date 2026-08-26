import os


def increment_path(
    path: str | os.PathLike[str],
    exist_ok: bool = False,
    sep: str = "",
) -> str | os.PathLike[str]:
    """Create a directory path, appending an increment when it already exists."""
    if os.path.exists(path) and not exist_ok:
        base, suffix = os.path.splitext(path) if os.path.isfile(path) else (path, "")

        for n in range(2, 9999):
            incremented_path = f"{base}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(incremented_path):
                path = incremented_path
                break

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)  # make directory

    return path
