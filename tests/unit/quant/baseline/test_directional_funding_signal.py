from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.research.baseline.signal import (
    _settled_funding_rates,
    generate_directional_funding_signals,
)
from src.research.contracts import StrategySpec

_SPEC = StrategySpec(entry_period=5, exit_period=3, ema_period=5, atr_period=5)


def _directional_frame() -> pd.DataFrame:
    """Flat base with an isolated long breakout at 260 and mirror breakdown at 400."""
    n = 500
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    o = np.full(n, 100.0)
    h = np.full(n, 101.0)
    l_ = np.full(n, 99.0)
    c = np.full(n, 100.0)

    o[260] = 106.0
    h[260] = 108.0
    l_[260] = 105.0
    c[260] = 107.0
    for t in range(261, 268):
        o[t] = 106.0
        h[t] = 107.0
        l_[t] = 105.0
        c[t] = 106.0

    o[400] = 94.0
    h[400] = 96.0
    l_[400] = 92.0
    c[400] = 93.0
    for t in range(401, 408):
        o[t] = 94.0
        h[t] = 95.0
        l_[t] = 93.0
        c[t] = 94.0

    return pd.DataFrame({
        "open": o, "high": h, "low": l_, "close": c, "volume": 1000.0,
    }, index=idx)


def test_long_breakout_negative_funding_is_signal_only_at_next_open() -> None:
    # SC-SGV2-01: a settled negative funding rate does not suppress a long
    # breakout, and the signal is emitted at the completed decision bar so the
    # order executes no earlier than the next bar open.
    df = _directional_frame()
    funding = pd.Series(
        [-0.0005, -0.0004],
        index=[df.index[200], df.index[259]],
    )
    features = generate_directional_funding_signals(df, _SPEC, funding)

    assert features.loc[df.index[260], "long_entry_signal"]
    assert not features.loc[df.index[260], "short_entry_signal"]

    from src.research.baseline.backtest import run_directional_backtest
    from src.research.contracts import CostModel

    result = run_directional_backtest(df, _SPEC, CostModel(), funding)
    long_trades = result.trades[result.trades["side"] == "long"]
    assert len(long_trades) >= 1
    entry_bar = int(long_trades["entry_bar"].iloc[0])
    assert entry_bar == 261, "signal at bar 260 must fill at bar 261 open"


def test_future_funding_cannot_change_prior_signal() -> None:
    # SC-SGV2-02: a positive funding event published after a decision bar cannot
    # veto the earlier long breakout (only settled funding <= bar close is read).
    df = _directional_frame()
    future_positive = pd.Series([0.001], index=[df.index[400]])

    features = generate_directional_funding_signals(df, _SPEC, future_positive)
    assert not features.loc[df.index[260], "long_entry_signal"]
    assert not features.loc[df.index[260], "short_entry_signal"]

    future_negative = pd.Series([-0.001], index=[df.index[400]])
    gated = generate_directional_funding_signals(df, _SPEC, future_negative)
    assert not gated.loc[df.index[260], "long_entry_signal"]


def test_short_breakdown_positive_funding_signals_short() -> None:
    # SC-SGV2-03 (signal surface): a mirror breakdown with settled funding >= 0
    # emits a short signal and never a long signal.
    df = _directional_frame()
    funding = pd.Series(
        [0.0005, 0.0006],
        index=[df.index[200], df.index[399]],
    )
    features = generate_directional_funding_signals(df, _SPEC, funding)
    assert features.loc[df.index[400], "short_entry_signal"]
    assert not features.loc[df.index[400], "long_entry_signal"]


class TestFailClosedFunding:
    def test_empty_bar_index_is_rejected(self) -> None:
        with pytest.raises(DataIntegrityError, match="non-empty DatetimeIndex"):
            _settled_funding_rates(
                pd.Series([0.001], index=pd.date_range("2024-01-01", periods=1, tz="UTC")),
                pd.DatetimeIndex([], tz="UTC"),
            )

    def test_missing_settled_funding_disables_component(self) -> None:
        # SC-SGV2-04: no settled event at or before the bar means no entry,
        # never a zero-filled signal.
        df = _directional_frame()
        funding = pd.Series([-0.0005], index=[df.index[259]])
        features = generate_directional_funding_signals(df, _SPEC, funding)
        assert not features.loc[df.index[200], "long_entry_signal"]
        assert not features.loc[df.index[200], "short_entry_signal"]

    def test_non_finite_funding_raises(self) -> None:
        df = _directional_frame()
        funding = pd.Series([np.nan], index=[df.index[259]])
        with pytest.raises(DataIntegrityError, match="finite"):
            generate_directional_funding_signals(df, _SPEC, funding)

    def test_non_monotonic_funding_raises(self) -> None:
        df = _directional_frame()
        funding = pd.Series(
            [-0.0005, -0.0004],
            index=[df.index[259], df.index[200]],
        )
        with pytest.raises(DataIntegrityError, match="monotonic"):
            generate_directional_funding_signals(df, _SPEC, funding)

    def test_duplicate_funding_raises(self) -> None:
        df = _directional_frame()
        funding = pd.Series(
            [-0.0005, -0.0004],
            index=[df.index[259], df.index[259]],
        )
        with pytest.raises(DataIntegrityError, match="duplicate"):
            generate_directional_funding_signals(df, _SPEC, funding)

    def test_empty_funding_raises(self) -> None:
        df = _directional_frame()
        with pytest.raises(DataIntegrityError, match="non-empty"):
            generate_directional_funding_signals(df, _SPEC, pd.Series(dtype=float))

    def test_invalid_funding_index_raises(self) -> None:
        df = _directional_frame()
        funding = pd.Series([-0.0005], index=["not-a-date"])
        with pytest.raises(DataIntegrityError, match="datetimes"):
            generate_directional_funding_signals(df, _SPEC, funding)
