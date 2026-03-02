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
    holdout_ratio: float = 0.30,
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

    # [NEW] Priority 4: Rolling Window (Non-Expanding) Purged Walk-Forward
    # 확장형(Expanding)의 치명적 단점인 '과거 레짐 편향(Memory Bias)'을 제거하기 위해,
    # 훈련 윈도우의 크기를 고정하고 슬라이딩(Sliding)시켜 항상 '최근 장세'에만 적응하도록 강제합니다.
    # [INSTITUTIONAL] 분기별 롤링(Quarterly Rolling)을 위한 훈련/테스트 황금비율
    # 전체 CV 구간의 60%(약 1년)를 훈련에 사용하고, 나머지 40%를 N개의 테스트(분기)로 분할합니다.
    train_size_ratio = 0.6
    train_size: int = int(train_bars_total * train_size_ratio)
    test_size: int = (train_bars_total - train_size) // n_folds
    stride: int = test_size

    if test_size == 0 or train_size == 0:
        _logger.warning(
            "Data too small: train_bars_total=%d, n_folds=%d. Skipping CV folds.",
            train_bars_total, n_folds,
        )
        return [], holdout_fold

    splits: List[CVFold] = []

    for i in range(n_folds):
        # 훈련 시작점을 계속 전진시켜 오래된 데이터를 망각(Forgetting)함
        train_start: int = i * stride
        train_end: int = train_start + train_size
        test_start: int = train_end + embargo

        # 마지막 폴드는 남은 자투리 데이터를 모두 포함하도록 보정
        test_end: int = train_end + test_size if i < n_folds - 1 else train_bars_total

        # Guard: skip degenerate folds where embargo collapses the test window.
        if test_start < test_end:
            splits.append((train_start, train_end, test_start, test_end))   
    return splits, holdout_fold
