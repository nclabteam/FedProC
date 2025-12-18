import os


def increment_path(path, exist_ok=False, sep=""):
    """
    Generates an incremented file or directory path if it exists, always making the directory; args: path, exist_ok=False, sep="".

    Example: runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc
    """
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
