"""
Pure metric functions for federated learning experiment analysis.

All functions take a list of floats and return a scalar or None.
Zero I/O, zero dependencies beyond stdlib + numpy.
"""

from collections import Counter
from typing import List, Literal, Optional, Tuple

import numpy as np

AGG_MODES = ("min", "max", "mean", "last", "median")
AggMode = Literal["min", "max", "mean", "last", "median"]


def compute_per_run_agg(
    vals: List[float], mode: AggMode = "min"
) -> Optional[float]:
    """Compute a single aggregate value from a list of values."""
    if not vals:
        return None
    if mode == "min":
        return min(vals)
    elif mode == "max":
        return max(vals)
    elif mode == "mean":
        return float(np.mean(vals))
    elif mode == "last":
        return vals[-1]
    elif mode == "median":
        return float(np.median(vals))
    return None


def last_improvement_round(vals: List[float]) -> Optional[float]:
    """Compute the last round where loss improved."""
    if len(vals) < 2:
        return None
    best_so_far = vals[0]
    last_imp_idx = 0
    for i in range(1, len(vals)):
        if vals[i] < best_so_far:
            last_imp_idx = i + 1  # 1-based
            best_so_far = vals[i]
    return float(last_imp_idx) if last_imp_idx > 0 else None


def improvement_streaks(
    vals: List[float],
) -> Tuple[Optional[float], Optional[float]]:
    """Compute longest and most frequent improvement streaks.

    Returns:
        Tuple of (longest_streak, most_frequent_streak)
    """
    if len(vals) < 2:
        return None, None

    best_so_far = vals[0]
    improvements: List[bool] = []
    for i in range(1, len(vals)):
        if vals[i] < best_so_far:
            improvements.append(True)
            best_so_far = vals[i]
        else:
            improvements.append(False)

    # Compute streaks
    max_streak = 0
    cur_streak = 0
    streaks: List[int] = []
    for imp in improvements:
        if imp:
            cur_streak += 1
        else:
            if cur_streak > 0:
                streaks.append(cur_streak)
            max_streak = max(max_streak, cur_streak)
            cur_streak = 0
    if cur_streak > 0:
        streaks.append(cur_streak)
        max_streak = max(max_streak, cur_streak)

    longest = float(max_streak) if max_streak > 0 else None

    # Most frequent
    most_frequent = None
    if streaks:
        cnt = Counter(streaks)
        most_common = cnt.most_common()
        max_freq = most_common[0][1]
        candidates = [length for length, freq in most_common if freq == max_freq]
        most_frequent = float(max(candidates))

    return longest, most_frequent


def oscillation_count(vals: List[float]) -> Optional[float]:
    """Compute number of direction changes in loss."""
    if len(vals) < 2:
        return None
    deltas = [vals[i] - vals[i - 1] for i in range(1, len(vals))]
    signs = [0 if d == 0 else (1 if d > 0 else -1) for d in deltas if d != 0]
    if len(signs) < 2:
        return 0.0
    oscillations = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
    return float(oscillations)


def improvement_ratio(vals: List[float]) -> Optional[float]:
    """Compute fraction of rounds with improvement."""
    if len(vals) < 2:
        return None
    best_so_far = vals[0]
    improvements = 0
    for val in vals[1:]:
        if val < best_so_far:
            improvements += 1
            best_so_far = val
    return float(improvements) / float(len(vals) - 1)


def improvement_magnitude(vals: List[float]) -> Optional[float]:
    """Compute average magnitude of loss reduction between improvements."""
    if len(vals) < 2:
        return None
    last_impr_loss: List[float] = []
    best_so_far = vals[0]
    for val in vals:
        if val < best_so_far:
            last_impr_loss.append(val)
            best_so_far = val
    run_deltas = []
    for i in range(1, len(last_impr_loss)):
        delta = last_impr_loss[i - 1] - last_impr_loss[i]
        if delta > 0:
            run_deltas.append(delta)
    return float(np.mean(run_deltas)) if run_deltas else None
