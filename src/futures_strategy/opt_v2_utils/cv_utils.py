"""
교차 검증(Cross-Validation)을 위해 데이터를 고정된 훈련 세트와 이후의 테스트 세트로 나누는 기능을 제공함.
시계열 데이터의 과거 학습과 미래 검증 구간 분리를 위한 로직을 포함함.
"""
import pandas as pd
from typing import List, Tuple

def build_anchored_folds(df: pd.DataFrame, n_folds: int = 3, embargo: int = 0) -> List[Tuple[int, int, int]]:
    """Build train(0~idx) -> OOS(idx~idx_next) splits."""
    n_bars: int = len(df)
    if n_bars < 500:
        return []

    block: int = n_bars // (n_folds + 1)
    splits: List[Tuple[int, int, int]] = []
    
    for i in range(1, n_folds + 1):
        train_end: int = block * i
        test_start: int = train_end + embargo
        test_end: int = (block * (i + 1)) if i < n_folds else n_bars

        if test_start < test_end:
            splits.append((train_end, test_start, test_end))

    return splits
