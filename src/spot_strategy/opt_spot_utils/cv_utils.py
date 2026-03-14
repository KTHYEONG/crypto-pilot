"""
교차 검증(Cross-Validation)을 위해 데이터를 고정된 훈련 세트와 이후의 테스트 세트로 나누는 기능을 제공함.
시계열 데이터의 과거 학습과 미래 검증 구간 분리를 위한 로직을 포함함.
"""
import logging
import pandas as pd
from typing import List, Tuple

_logger: logging.Logger = logging.getLogger("opt_spot")

# CV folds  : 4-tuple (train_start, train_end, test_start, test_end)
# Holdout fold: 3-tuple (train_end, test_start, test_end)
CVFold = Tuple[int, int, int, int]
HoldoutFold = Tuple[int, int, int]


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
