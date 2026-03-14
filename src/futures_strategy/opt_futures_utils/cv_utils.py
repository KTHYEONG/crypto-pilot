"""
교차 검증(Cross-Validation)을 위해 데이터를 고정된 훈련 세트와 이후의 테스트 세트로 나누는 기능을 제공함.
시계열 데이터의 과거 학습과 미래 검증 구간 분리를 위한 로직을 포함함.
"""
import logging
import pandas as pd
from typing import List, Tuple

_logger: logging.Logger = logging.getLogger("opt_futures")

# CV folds  : 4-tuple (train_start, train_end, test_start, test_end)
# Holdout fold: 3-tuple (train_end, test_start, test_end)  ← evaluator unpacks via len() check
CVFold = Tuple[int, int, int, int]
HoldoutFold = Tuple[int, int, int]


def build_purged_walk_forward_folds(
    df: pd.DataFrame,
    n_folds: int = 6,
    holdout_ratio: float = 0.20,
    embargo: int = 0,
) -> Tuple[List[CVFold], HoldoutFold]:
    """
    [CRITICAL FIX FOR TREND FOLLOWING]
    Builds a single continuous In-Sample (IS) block for evaluation.
    Chopping the timeline artificially cuts holding periods and forces arbitrary
    position closures, which completely corrupts Trailing Stop metrics in Optuna.
    Optuna must evaluate the entire dataset continuously.
    """
    # 1. 시계열 교차 검증 (4-Fold Walk-Forward)
    # 데이터를 시간순으로 4개 구간으로 나누어 점진적으로 학습/검증 범위를 확장함.
    n_bars: int = len(df)
    if n_bars < 500:
        return [], (0, 0, 0)

    # 20%는 최종 Holdout(검증)으로 제외하고, 나머지 80% 내에서 n_folds 구성을 위해 인덱스 계산
    is_bars = int(n_bars * (1 - holdout_ratio))
    
    # 각 폴드의 테스트 구간 크기 계산 (Embargo 고려)
    # 총 IS 범위 내에서 n_folds개의 테스트 구간이 들어갈 수 있도록 조정
    # [is_bars = (fold_size * n_folds) + total_embargo_gaps] 가 되도록 설계
    usable_is_bars = is_bars - (n_folds * embargo)
    if usable_is_bars <= 0: return [], (0, 0, 0) # Data too short
    
    fold_size = usable_is_bars // n_folds
    if fold_size < 10: return [], (0, 0, 0) # Each fold must be meaningful

    splits: List[CVFold] = []
    for i in range(1, n_folds + 1):
        # 학습 종료 지점 산출
        train_end = i * fold_size + (i - 1) * embargo
        test_start = train_end + embargo
        # 마지막 폴드라도 is_bars를 넘지 않도록 강제 (Leakage 방지)
        test_end = min(is_bars, test_start + fold_size)
        
        if test_end > test_start:
            splits.append((0, train_end, test_start, test_end))

    # 최종 검증 구간 (Holdout) - CV에서 단 한 번도 보지 못한 미래 데이터
    ho_start = min(n_bars, is_bars + embargo)
    holdout_fold: HoldoutFold = (is_bars, ho_start, n_bars)

    return splits, holdout_fold
