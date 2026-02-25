"""
교차 검증(Cross-Validation)을 위해 데이터를 고정된 훈련 세트와 이후의 테스트 세트로 나누는 기능을 제공함.
시계열 데이터의 과거 학습과 미래 검증 구간 분리를 위한 로직을 포함함.
"""
import pandas as pd
from typing import List, Tuple

def build_anchored_folds(df: pd.DataFrame, n_folds: int = 3, holdout_ratio: float = 0.25, embargo: int = 0) -> Tuple[List[Tuple[int, int, int]], Tuple[int, int, int]]:
    """Build train(0~idx) -> Test(idx~idx_next) splits for CV, and one OOS Hold-out split."""
    n_bars: int = len(df)
    if n_bars < 500:
        return [], (0, 0, 0)

    train_bars: int = int(n_bars * (1.0 - holdout_ratio))
    holdout_start: int = train_bars + embargo
    holdout_fold: Tuple[int, int, int] = (train_bars, holdout_start, n_bars)

    block: int = train_bars // (n_folds + 1)
    splits: List[Tuple[int, int, int]] = []
    
    for i in range(1, n_folds + 1):
        train_end: int = block * i
        test_start: int = train_end + embargo
        test_end: int = (block * (i + 1)) if i < n_folds else train_bars

        if test_start < test_end:
            splits.append((train_end, test_start, test_end))

    return splits, holdout_fold
