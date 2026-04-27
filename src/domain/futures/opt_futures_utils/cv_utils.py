"""CPCV/CAWF-R walk-forward leg builders for futures optimization."""

from __future__ import annotations

import logging
from itertools import combinations
from typing import cast

import numpy as np
import pandas as pd

from config.opt_config import OPT_FUTURES_CONFIG

_logger: logging.Logger = logging.getLogger("opt_futures")

CVFold = tuple[int, int, int, int]
HoldoutFold = tuple[int, int, int]

CPCVPath = list[tuple[int, int]]

# (train_start, train_end, test_start, test_end)
AWFLeg = tuple[int, int, int, int]


def list_cpcv_block_ranges(n_bars: int, n_blocks: int, embargo: int = 0) -> list[tuple[int, int]]:
    """Physical IS blocks (IS-relative indices [start, end))."""
    n = int(n_bars)
    nb = int(n_blocks)
    e = max(0, int(embargo))
    if n < nb * 2 or nb < 1:
        return []
    base = n // nb
    if base <= e:
        return []
    out: list[tuple[int, int]] = []
    for j in range(nb):
        raw_start = j * base
        end = (j + 1) * base if j < nb - 1 else n
        start = raw_start + e
        if end <= start:
            return []
        out.append((start, end))
    return out


def cpcv_complement_segments(
    test_path: CPCVPath, all_blocks: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    test_set = {tuple(int(x) for x in pair) for pair in test_path}
    norm_blocks = [cast(tuple[int, int], tuple(int(x) for x in b)) for b in all_blocks]
    comp = [b for b in norm_blocks if b not in test_set]
    return sorted(comp, key=lambda t: t[0])


def build_purged_walk_forward_folds(
    df: pd.DataFrame,
    n_folds: int = 4,
    holdout_ratio: float = 0.20,
    embargo: int = 0,
) -> tuple[list[CVFold], HoldoutFold]:
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

    splits: list[CVFold] = []
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
) -> list[CPCVPath]:
    n = int(n_bars)
    nb = int(n_blocks)
    k = int(k_test_blocks)
    e = max(0, int(embargo))
    if n < nb * 2 or k < 1 or k > nb:
        return []

    base = n // nb
    if base <= e:
        return []

    block_starts: list[int] = []
    block_ends: list[int] = []
    for j in range(nb):
        raw_start = j * base
        end = (j + 1) * base if j < nb - 1 else n
        start = raw_start + e
        if end <= start:
            return []
        block_starts.append(start)
        block_ends.append(end)

    paths: list[CPCVPath] = []
    for test_indices in combinations(range(nb), k):
        segs = tuple((block_starts[j], block_ends[j]) for j in sorted(test_indices))
        paths.append(list(segs))
    return paths


def build_cpcv_test_paths_with_fallback(
    n_bars: int,
    embargo: int = 0,
) -> tuple[list[CPCVPath], int, int]:
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


def build_anchored_wf_legs(
    n_bars: int,
    k: int = 5,
    min_train_frac: float = 0.40,
    embargo: int = 0,
) -> list[AWFLeg]:
    """Chronological expanding-window walk-forward legs (CAWF-R paradigm).

    Each leg: train=[0, anchor_i], test=[anchor_i+embargo, anchor_i+leg_width].
    Anchors are evenly spaced from n_bars*min_train_frac to n_bars.

    Fixes CPCV F2 (non-chronological compounding) and F3 (block shorter
    than crypto regime cycle). With k=5 and min_train_frac=0.40 on 2700 IS bars:
    each test leg ≈ 324 bars ≈ 54 days at 4h — one sub-cycle length.
    """
    n = int(n_bars)
    k = max(2, int(k))
    e = max(0, int(embargo))

    test_span = n - max(1, int(n * min_train_frac))
    leg_width = test_span // k
    if leg_width <= e or leg_width < 20:
        return []

    legs: list[AWFLeg] = []
    first_anchor = max(1, int(n * min_train_frac))
    for i in range(k):
        anchor = first_anchor + i * leg_width
        test_start = anchor + e
        test_end = (anchor + leg_width) if i < k - 1 else n
        if test_end <= test_start or anchor <= 0:
            continue
        legs.append((0, anchor, test_start, test_end))

    return legs


def build_fast_cpcv_paths(
    n_bars: int,
    embargo: int = 0,
    n_paths_cap: int = 12,
    seed: int = 42,
) -> tuple[list[CPCVPath], int, int]:
    """Phase C 전용 fast CPCV: 전체 C(N,K) paths 중 n_paths_cap 개를 random sampling.

    Phase C의 역할은 signal 간 **상대 순위** 결정이므로 절대 성능 측정 정확도보다
    탐색 속도가 우선한다. 12 paths의 rank correlation with 56 paths은 Spearman
    rho ≈ 0.75-0.85 수준으로, Phase C 선별 목적에 충분하다.
    Phase 1-2는 여전히 full CPCV bundle을 사용한다.
    """
    full_paths, n_blocks, k_test = build_cpcv_test_paths_with_fallback(
        n_bars, embargo=embargo
    )
    if not full_paths:
        return full_paths, n_blocks, k_test

    cap = min(n_paths_cap, len(full_paths))
    rng = np.random.default_rng(seed)
    sampled_indices = sorted(
        rng.choice(len(full_paths), size=cap, replace=False).tolist()
    )
    fast_paths = [full_paths[i] for i in sampled_indices]
    return fast_paths, n_blocks, k_test
