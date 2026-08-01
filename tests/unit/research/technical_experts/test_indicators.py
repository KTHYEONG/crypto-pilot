from __future__ import annotations

import pandas as pd

from src.research.technical_experts.indicators import ema


def test_ema_is_causal_and_preserves_index() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="4h", tz="UTC")
    close = pd.Series([100.0, 101.0, 99.0, 102.0], index=index)

    result = ema(close, 2)

    assert result.index.equals(index)
    assert result.iloc[0] == close.iloc[0]
    assert result.iloc[-1] != close.iloc[-1]
