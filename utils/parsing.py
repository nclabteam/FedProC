def str2bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None

    normalized = str(value).strip().lower()
    if normalized in {"true", "t", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "f", "0", "no", "n", "off"}:
        return False

    raise ValueError(
        f"Invalid boolean value {value!r}. Use true/false, yes/no, on/off, or 1/0."
    )
