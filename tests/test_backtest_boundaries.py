from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.domain.futures.engine_fast_futures import BacktestEngineFast
from src.domain.futures.opt_futures_utils.evaluator import _segment_with_context


class _DummyStrategy:
    def __init__(self) -> None:
        self.name = "dummy"
        self.params = {
            "ATR_MULTIPLIER": 2.0,
            "TRAILING_MULTIPLIER": 2.0,
            "LEVERAGE": 1,
            "RISK_PER_TRADE": 0.01,
            "USE_COMPOUNDING": True,
        }

    def get_required_warmup(self, freq: str = "hourly") -> int:
        _ = freq
        return 0

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


def test_segment_with_context_uses_one_prior_bar() -> None:
    df = pd.DataFrame({"x": [10, 20, 30, 40]})
    segment, execution_start_idx = _segment_with_context(df, exec_start_idx=2, exec_end_idx=4)

    assert segment["x"].tolist() == [20, 30, 40]
    assert execution_start_idx == 1


def test_segment_with_context_skips_absolute_first_bar_when_no_prior_context() -> None:
    df = pd.DataFrame({"x": [10, 20, 30]})
    segment, execution_start_idx = _segment_with_context(df, exec_start_idx=0, exec_end_idx=3)

    assert segment["x"].tolist() == [10, 20, 30]
    assert execution_start_idx == 1


def test_engine_respects_execution_start_idx() -> None:
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime([0, 4 * 3600], unit="s"),
            "timestamp": np.array([0, 14_400_000], dtype=np.int64),
            "open": [100.0, 106.0],
            "high": [110.0, 108.0],
            "low": [99.0, 105.0],
            "close": [105.0, 107.0],
            "volume": [1.0, 1.0],
            "entry_upper": [0.0, 0.0],
            "entry_lower": [np.finfo(np.float64).max, np.finfo(np.float64).max],
            "trend_direction": [1, 1],
            "strength_filter": [1, 1],
            "atr": [1.0, 1.0],
            "funding_rate_sum": [0.0, 0.0],
            "daily_macro_ema": [100.0, 100.0],
        }
    )

    engine = BacktestEngineFast(
        hourly_df=df,
        daily_df=df,
        strategy=_DummyStrategy(),
        initial_balance=800.0,
        execution_start_idx=1,
    )
    result = engine.run()

    assert result["total_trades"] == 1
    assert result["trades_df"]["entry_time"].iloc[0] == df["datetime"].iloc[1]
