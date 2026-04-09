"""
Cross-validation helpers: purged walk-forward (legacy) and CPCV test paths for spot optimization.
"""
from __future__ import annotations

import logging
from itertools import combinations
from typing import List, Tuple

import pandas as pd

from config.opt_config import OPT_SPOT_CONFIG

_logger: logging.Logger = logging.getLogger("opt_spot")

# CV folds  : 4-tuple (train_start, train_end, test_start, test_end)
# Holdout fold: 3-tuple (train_end, test_start, test_end)
CVFold = Tuple[int, int, int, int]
HoldoutFold = Tuple[int, int, int]

# CPCV: each path is a list of disjoint test segment index ranges [start, end)
CPCVPath = List[Tuple[int, int]]


def list_cpcv_block_ranges(n_bars: int, n_blocks: int, embargo: int = 0) -> List[Tuple[int, int]]:
    """
    Physical IS blocks used by build_cpcv_test_paths (same geometry, IS-relative indices).
    """
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
    """Train/CPCV complement = all physical blocks not selected as test in this path."""
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
    n_bars: int = len(df)
    if n_bars < 500:
        return [], (0, 0, 0)

    # IS(In-Sample) 영역 계산
    is_bars = int(n_bars * (1 - holdout_ratio))
    
    # 엠바고를 제외한 순수 사용 가능 IS 구간
    usable_is_bars = is_bars - (n_folds * embargo)
    if usable_is_bars <= 0: return [], (0, 0, 0)
    
    fold_size = usable_is_bars // n_folds
    if fold_size < 10: return [], (0, 0, 0)

    splits: List[CVFold] = []
    for i in range(1, n_folds + 1):
        # 정교한 Expanding Window + Embargo 처리
        train_end = i * fold_size + (i - 1) * embargo
        test_start = train_end + embargo
        # IS 경계를 절대 넘지 않도록 제한 (Leakage 방지)
        test_end = min(is_bars, test_start + fold_size)
        
        if test_end > test_start:
            splits.append((0, train_end, test_start, test_end))

    # Holdout: CV에서 물리적으로 격리된 완전한 미래 데이터
    ho_start = min(n_bars, is_bars + embargo)
    holdout_fold: HoldoutFold = (is_bars, ho_start, n_bars)

    return splits, holdout_fold


def build_cpcv_test_paths(
    n_bars: int,
    n_blocks: int,
    k_test_blocks: int,
    embargo: int = 0,
) -> List[CPCVPath]:
    """
    Combinatorial purged CV: choose k_test_blocks disjoint blocks as test; each path is their union.

    Each physical block [j*base, end_j) is trimmed to [j*base+embargo, end_j) so the first `embargo`
    bars of each candidate test block are excluded (train/test boundary mitigation).
    """
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
    """Prefer N/K from OPT_SPOT_CONFIG (default 8/3); fallback 6/2 then 4/2 if empty."""
    n_primary = int(OPT_SPOT_CONFIG.get("CPCV_N_BLOCKS", 8))
    k_primary = int(OPT_SPOT_CONFIG.get("CPCV_K_TEST", 3))
    paths = build_cpcv_test_paths(n_bars, n_primary, k_primary, embargo=embargo)
    if paths:
        return paths, n_primary, k_primary
    paths_fb62 = build_cpcv_test_paths(n_bars, 6, 2, embargo=embargo)
    if paths_fb62:
        return paths_fb62, 6, 2
    paths_fb = build_cpcv_test_paths(n_bars, 4, 2, embargo=embargo)
    return paths_fb, 4, 2