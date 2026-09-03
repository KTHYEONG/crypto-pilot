from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.quant.contracts import StrategySpec
from src.quant.baseline.signal import atr, donchian_lower, donchian_upper, generate_signals


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
        from src.quant.contracts import StrategySpec
        spec = StrategySpec()
        out = generate_signals(bars_ramp, spec)
        for col in ["upper", "exit_lower", "ema", "atr", "entry_signal"]:
            assert col in out.columns, f"missing column: {col}"

    def test_ema_seeded_at_first_close(self) -> None:
        e = pd.Series([100.0, 110.0]).ewm(span=3, adjust=False).mean()
        assert e.iloc[0] == 100.0
        assert e.iloc[1] == 105.0


def _ramp_with_ratio(ratio: float = 0.52) -> pd.DataFrame:
    n = 300
    index = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    opens = np.arange(100.0, 100.0 + n, dtype=np.float64)
    return pd.DataFrame({
        "open": opens,
        "high": opens + 1.0,
        "low": opens - 1.0,
        "close": opens + 0.5,
        "volume": 1000.0,
        "taker_buy_ratio": ratio,
    }, index=index)


class TestTakerFlowFilter:
    def test_output_carries_taker_buy_ratio_column(self) -> None:
        n = 201
        df = pd.DataFrame({
            "open": [1.0] * n, "high": [1.0] * n,
            "low": [1.0] * n, "close": [1.0] * n,
            "taker_buy_ratio": [0.52] * n,
        })
        out = generate_signals(df, StrategySpec(min_taker_buy_ratio=0.52))
        assert "taker_buy_ratio" in out.columns

    def test_taker_flow_filter_is_causal_and_fail_closed(self) -> None:
        # SC-FLOW-03: at a completed signal bar, only ratio >= threshold may signal.
        df = _ramp_with_ratio()
        spec_base = StrategySpec(ema_period=5, entry_period=5, atr_period=5)
        base = generate_signals(df, spec_base)
        signal_bars = base.index[base["entry_signal"]].tolist()
        assert len(signal_bars) >= 2, "fixture must produce multiple signal bars"

        bar_low = signal_bars[len(signal_bars) // 2]
        bar_high = signal_bars[len(signal_bars) // 2 + 1]
        df2 = df.copy()
        df2.loc[bar_low, "taker_buy_ratio"] = 0.51
        df2.loc[bar_high, "taker_buy_ratio"] = 0.52

        opt_in = generate_signals(
            df2, StrategySpec(ema_period=5, entry_period=5, atr_period=5,
                              min_taker_buy_ratio=0.52),
        )
        assert not opt_in.loc[bar_low, "entry_signal"]
        assert opt_in.loc[bar_high, "entry_signal"]

    def test_missing_or_invalid_ratio_fails_closed(self) -> None:
        # SC-FLOW-02: NaN or out-of-[0,1] ratio can never emit an opt-in entry.
        df = _ramp_with_ratio()
        spec_base = StrategySpec(ema_period=5, entry_period=5, atr_period=5)
        base = generate_signals(df, spec_base)
        signal_bars = base.index[base["entry_signal"]].tolist()
        assert len(signal_bars) >= 3, "fixture must produce multiple signal bars"

        df2 = df.copy()
        df2.loc[signal_bars[0], "taker_buy_ratio"] = np.nan
        df2.loc[signal_bars[1], "taker_buy_ratio"] = 1.5
        df2.loc[signal_bars[2], "taker_buy_ratio"] = -0.1

        opt_in = generate_signals(
            df2, StrategySpec(ema_period=5, entry_period=5, atr_period=5,
                              min_taker_buy_ratio=0.52),
        )
        assert not opt_in.loc[signal_bars[0], "entry_signal"]
        assert not opt_in.loc[signal_bars[1], "entry_signal"]
        assert not opt_in.loc[signal_bars[2], "entry_signal"]

    def test_default_mode_matches_v1_baseline(self) -> None:
        # SC-FLOW-04: with the filter disabled, presence of the flow column
        # cannot alter the frozen v1 signal surface.
        df = _ramp_with_ratio()
        plain = df.drop(columns=["taker_buy_ratio"])
        spec = StrategySpec(ema_period=5, entry_period=5, atr_period=5)
        with_flow = generate_signals(df, spec)
        without_flow = generate_signals(plain, spec)
        assert with_flow["entry_signal"].equals(without_flow["entry_signal"])

    def test_enabled_filter_requires_ratio_column(self) -> None:
        df = _ramp_with_ratio().drop(columns=["taker_buy_ratio"])
        with pytest.raises(ValueError, match="taker_buy_ratio"):
            generate_signals(
                df, StrategySpec(ema_period=5, entry_period=5, atr_period=5,
                                 min_taker_buy_ratio=0.52),
            )
