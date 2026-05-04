"""
I/O helpers and unit conversion for analysis modules.

Shared by single.py, multi.py, landscape.py, and other analysis tools.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl

logger = logging.getLogger(__name__)

# Sentinel value for missing data in server/client CSVs
DEFAULT_VALUE: float = 9_999_999.0

# Unit options
TIME_UNITS = ("s", "ms", "m", "h")
SIZE_UNITS = ("b", "kb", "mb", "gb", "tb")

# Time columns (base unit: seconds)
TIME_COLUMNS = ("efficiency", "time_per_experiment")

# Size columns (base unit: MB)
SIZE_COLUMNS = ("downlink", "uplink", "communication")


# =============================================================================
# Unit conversion
# =============================================================================


def convert_time(value: float, from_unit: str = "s", to_unit: str = "s") -> float:
    """Convert time value between units (s, ms, m, h)."""
    to_seconds = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    from_seconds = {"ms": 1000.0, "s": 1.0, "m": 1.0 / 60.0, "h": 1.0 / 3600.0}
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()
    if from_unit not in to_seconds:
        raise ValueError(f"Unknown time unit: {from_unit}")
    if to_unit not in from_seconds:
        raise ValueError(f"Unknown time unit: {to_unit}")
    return value * to_seconds[from_unit] * from_seconds[to_unit]


def convert_size(value: float, from_unit: str = "mb", to_unit: str = "mb") -> float:
    """Convert size value between units (b, kb, mb, gb, tb)."""
    to_bytes = {
        "b": 1.0,
        "kb": 1024.0,
        "mb": 1024.0**2,
        "gb": 1024.0**3,
        "tb": 1024.0**4,
    }
    from_bytes = {
        "b": 1.0,
        "kb": 1.0 / 1024.0,
        "mb": 1.0 / (1024.0**2),
        "gb": 1.0 / (1024.0**3),
        "tb": 1.0 / (1024.0**4),
    }
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()
    if from_unit not in to_bytes:
        raise ValueError(f"Unknown size unit: {from_unit}")
    if to_unit not in from_bytes:
        raise ValueError(f"Unknown size unit: {to_unit}")
    return value * to_bytes[from_unit] * from_bytes[to_unit]


# =============================================================================
# File I/O
# =============================================================================


def read_json(path: Path) -> Optional[Dict]:
    """Read a JSON file and return its contents, or None on failure."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.debug("Failed to load JSON: %s", path)
        return None


def read_csv(path: Path, max_lines: Optional[int] = None) -> Optional[Dict[str, List]]:
    """Read a CSV file and return as dict of lists, or None on failure."""
    if not path.exists():
        return None
    try:
        df = pl.read_csv(path, n_rows=max_lines) if max_lines else pl.read_csv(path)
        return df.to_dict(as_series=False)
    except Exception as e:
        logger.warning("Failed to read CSV %s: %s", path, e)
        return None


def read_timing(path: Path) -> List[Dict]:
    """Read timing.json for an experiment."""
    data = read_json(path)
    return data if isinstance(data, list) else []


# =============================================================================
# Parsing
# =============================================================================


def parse_numeric_list(
    seq: List, exclude_zero: bool = False, sentinel: float = DEFAULT_VALUE
) -> List[float]:
    """Parse a list to extract valid numeric values, filtering sentinels."""
    vals: List[float] = []
    for x in seq:
        try:
            xv = float(x)
        except (ValueError, TypeError):
            continue
        if xv == sentinel:
            continue
        if exclude_zero and xv == 0:
            continue
        vals.append(xv)
    return vals


# =============================================================================
# Config helpers
# =============================================================================


def resolve_loss_metric(save_local_model: bool) -> str:
    """Resolve the loss metric name based on save_local_model flag."""
    return "personal_avg_test_loss" if save_local_model else "global_avg_test_loss"


def load_config(experiment_dir: Path) -> Dict:
    """Load config.json from an experiment directory."""
    config_path = experiment_dir / "config.json"
    data = read_json(config_path)
    return dict(data) if isinstance(data, dict) else {}
