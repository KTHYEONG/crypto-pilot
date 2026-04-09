"""
CPCV test paths for futures optimization (spot-aligned geometry, fresh-start-per-segment in evaluator).
"""
from __future__ import annotations

import logging
from itertools import combinations
from typing import List, Tuple

import pandas as pd

from config.opt_config import OPT_FUTURES_CONFIG

_logger: logging.Logger = logging.getLogger("opt_futures")

CVFold = Tuple[int, int, int, int]
HoldoutFold = Tuple[int, int, int]

CPCVPath = List[Tuple[int, int]]


def list_cpcv_block_ranges(n_bars: int, n_blocks: int, embargo: int = 0) -> List[Tuple[int, int]]:
    """Physical IS blocks (IS-relative indices [start, end))."""
    n = int(n_bars)
    nb = int(n_blocks)
    e = max(0, int(embargo))
    if n < nb * 2 or nb < 1:
        return []
    base = n // nb
    if base <= e:
        return []
    out: List[Tuple[int, int]] = []
    for j in range(nb):
        raw_start = j * base
        end = (j + 1) * base if j < nb - 1 else n
        start = raw_start + e
        if end <= start:
            return []
        out.append((start, end))
    return out


def cpcv_complement_segments(test_path: CPCVPath, all_blocks: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    test_set = {tuple(int(x) for x in pair) for pair in test_path}
    norm_blocks = [tuple(int(x) for x in b) for b in all_blocks]
    comp = [b for b in norm_blocks if b not in test_set]
    return sorted(comp, key=lambda t: t[0])


def build_purged_walk_forward_folds(
    df: pd.DataFrame,
    n_folds: int = 4,
    holdout_ratio: float = 0.20,
    embargo: int = 0,
) -> Tuple[List[CVFold], HoldoutFold]:
    """Legacy walk-forward splits (optional tooling / tests)."""
    n_bars: int = len(df)
    if n_bars < 500:
        return [], (0, 0, 0)

    is_bars = int(n_bars * (1 - holdout_ratio))
    usable_is_bars = is_bars - (n_folds * embargo)
    if usable_is_bars <= 0:
        return [], (0, 0, 0)

    fold_size = usable_is_bars // n_folds
    if fold_size < 10:
        return [], (0, 0, 0)

    splits: List[CVFold] = []
    for i in range(1, n_folds + 1):
        train_end = i * fold_size + (i - 1) * embargo
        test_start = train_end + embargo
        test_end = min(is_bars, test_start + fold_size)
        if test_end > test_start:
            splits.append((0, train_end, test_start, test_end))

    ho_start = min(n_bars, is_bars + embargo)
    holdout_fold: HoldoutFold = (is_bars, ho_start, n_bars)

    return splits, holdout_fold


def build_cpcv_test_paths(
    n_bars: int,
    n_blocks: int,
    k_test_blocks: int,
    embargo: int = 0,
) -> List[CPCVPath]:
    n = int(n_bars)
    nb = int(n_blocks)
    k = int(k_test_blocks)
    e = max(0, int(embargo))
    if n < nb * 2 or k < 1 or k > nb:
        return []

    base = n // nb
    if base <= e:
        return []

    block_starts: List[int] = []
    block_ends: List[int] = []
    for j in range(nb):
        raw_start = j * base
        end = (j + 1) * base if j < nb - 1 else n
        start = raw_start + e
        if end <= start:
            return []
        block_starts.append(start)
        block_ends.append(end)

    paths: List[CPCVPath] = []
    for test_indices in combinations(range(nb), k):
        segs = tuple((block_starts[j], block_ends[j]) for j in sorted(test_indices))
        paths.append(list(segs))
    return paths


def build_cpcv_test_paths_with_fallback(
    n_bars: int,
    embargo: int = 0,
) -> Tuple[List[CPCVPath], int, int]:
    """Prefer N/K from OPT_FUTURES_CONFIG; fallback 6/2 then 4/2."""
    n_primary = int(OPT_FUTURES_CONFIG.get("FUTURES_CPCV_N_BLOCKS", 8))
    k_primary = int(OPT_FUTURES_CONFIG.get("FUTURES_CPCV_K_TEST", 3))
    paths = build_cpcv_test_paths(n_bars, n_primary, k_primary, embargo=embargo)
    if paths:
        return paths, n_primary, k_primary
    paths_fb62 = build_cpcv_test_paths(n_bars, 6, 2, embargo=embargo)
    if paths_fb62:
        return paths_fb62, 6, 2
    paths_fb = build_cpcv_test_paths(n_bars, 4, 2, embargo=embargo)
    return paths_fb, 4, 2
