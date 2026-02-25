"""
교차 검증(Cross-Validation)을 위해 데이터를 고정된 훈련 세트와 이후의 테스트 세트로 나누는 기능을 제공함.
시계열 데이터의 과거 학습과 미래 검증 구간 분리를 위한 로직을 포함함.
"""
import logging
import pandas as pd
from typing import List, Tuple

_logger: logging.Logger = logging.getLogger("opt_v2")

# CV folds  : 4-tuple (train_start, train_end, test_start, test_end)
# Holdout fold: 3-tuple (train_end, test_start, test_end)  ← evaluator unpacks via len() check
CVFold = Tuple[int, int, int, int]
HoldoutFold = Tuple[int, int, int]


def build_purged_walk_forward_folds(
    df: pd.DataFrame,
    n_folds: int = 3,
    holdout_ratio: float = 0.25,
    embargo: int = 0,
) -> Tuple[List[CVFold], HoldoutFold]:
    """Build Purged Walk-Forward (PWF) splits for CV, and one OOS Hold-out split.

    Returns
    -------
    splits: List[CVFold]
        Each element is (train_start, train_end, test_start, test_end) — **4-tuple**.
        Evaluator detects length==4 and unpacks accordingly.
    holdout_fold: HoldoutFold
        (train_end, test_start, test_end) — **3-tuple** compatible with legacy unpack path.
        test_end == n_bars (uses train_bars_total as last test chunk for the last CV fold,
        and n_bars for the true hold-out set).
    """
    n_bars: int = len(df)
    if n_bars < 500:
        return [], (0, 0, 0)

    train_bars_total: int = int(n_bars * (1.0 - holdout_ratio))
    holdout_start: int = train_bars_total + embargo

    # Holdout test_end is the full dataset end (n_bars), not train_bars_total.
    holdout_fold: HoldoutFold = (train_bars_total, holdout_start, n_bars)

    # Calculate block sizes
    # e.g. n_folds=3 → total_chunks=6. Train window occupies 3 chunks, test 1 chunk (sliding).
    total_chunks: int = n_folds * 2
    chunk_size: int = train_bars_total // total_chunks

    if chunk_size == 0:
        _logger.warning(
            "chunk_size==0: train_bars_total=%d, total_chunks=%d. Skipping CV folds.",
            train_bars_total, total_chunks,
        )
        return [], holdout_fold

    # Sliding window indices
    # Fold k: Train C[k]..C[k+train_chunks_per_fold-1], Test C[k+train_chunks_per_fold]
    train_chunks_per_fold: int = total_chunks - n_folds

    splits: List[CVFold] = []

    for i in range(n_folds):
        train_start: int = i * chunk_size
        train_end: int = (i + train_chunks_per_fold) * chunk_size
        test_start: int = train_end + embargo
        # Last CV fold's test_end is train_bars_total (immediately before hold-out region).
        test_end: int = (
            (i + train_chunks_per_fold + 1) * chunk_size
            if i < n_folds - 1
            else train_bars_total
        )

        # Guard: skip degenerate folds where embargo collapses the test window.
        if test_start < test_end:
            splits.append((train_start, train_end, test_start, test_end))

    return splits, holdout_fold
