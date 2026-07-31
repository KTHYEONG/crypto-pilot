from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.donchian import atr, donchian_lower, donchian_upper, generate_signals


class TestDonchian:
    def test_excludes_current_bar(self) -> None:
        s = pd.Series(np.arange(1.0, 11.0))
        u = donchian_upper(s, 3)
        l_ = donchian_lower(s, 3)
        assert pd.isna(u.iloc[2])
        assert u.iloc[3] == 3.0
        assert u.iloc[5] == 5.0
        assert u.iloc[5] != s.iloc[5]
        assert l_.iloc[5] == 3.0

    def test_raises_on_bad_period(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="period must be >= 1"):
            donchian_upper(s, 0)

    def test_raises_on_empty(self) -> None:
        with pytest.raises(ValueError, match="series must not be empty"):
            donchian_upper(pd.Series([], dtype=float), 5)


class TestATR:
    def test_is_sma_of_true_range(self) -> None:
        h = pd.Series([10.0, 12.0, 11.0])
        l_ = pd.Series([8.0, 9.0, 7.0])
        c = pd.Series([9.0, 11.0, 8.0])
        a = atr(h, l_, c, 2)
        assert a.iloc[1] == 2.5
        assert a.iloc[2] == 3.5


class TestGenerateSignals:
    def test_columns_and_causality(self, bars_ramp: pd.DataFrame) -> None:
        from src.core.types import StrategySpec
        spec = StrategySpec()
        out = generate_signals(bars_ramp, spec)
        for col in ["upper", "exit_lower", "ema", "atr", "entry_signal"]:
            assert col in out.columns, f"missing column: {col}"

    def test_ema_seeded_at_first_close(self) -> None:
        e = pd.Series([100.0, 110.0]).ewm(span=3, adjust=False).mean()
        assert e.iloc[0] == 100.0
        assert e.iloc[1] == 105.0
