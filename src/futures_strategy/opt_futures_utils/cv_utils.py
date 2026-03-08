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
    n_bars: int = len(df)
    if n_bars < 500:
        return [], (0, 0, 0)

    # We evaluate 100% of the provided IS dataframe as a single run.
    # (The actual OOS is physically separated in main() via oos_data_maps)
    splits: List[CVFold] = [(0, 0, 0, n_bars)]
    
    # Dummy holdout fold (ignored since we process all data in the splits)
    holdout_fold: HoldoutFold = (n_bars, n_bars, n_bars)

    return splits, holdout_fold
