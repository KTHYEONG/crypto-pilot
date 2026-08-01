from __future__ import annotations

import numpy as np
import pytest

from src.research.contracts import CostModel
from src.research.oi_deleveraging.backtest import run_open_interest_deleveraging_screen

ZERO_COSTS = CostModel(fee_rate=0.0, slippage_rate=0.0)

_FALLING = [-0.01, -0.01, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]


def _run(data, costs: CostModel | None = None, delay: int = 1):
    return run_open_interest_deleveraging_screen(
        data, costs if costs is not None else CostModel(), signal_delay_bars=delay,
    )


class TestOIDeleveragingScreen:
    def test_open_interest_deleveraging_enters_only_on_fixed_state(
        self, make_oi_market_data,
    ) -> None:
        # FD-04: only the fixed deleveraging state (negative completed 24h mark
        # return AND negative daily OI-value change) holds a delayed short; the
        # rest of the window is cash.
        data = make_oi_market_data(
            n_bars=8,
            mark_return_24h=_FALLING,
            oi_change=[-1.0] * 8,
        )
        result = _run(data)
        assert len(result.trades) == 1
        trade = result.trades.iloc[0]
        assert trade["side"] == "short"
        assert trade["entry_bar"] == 2
        assert trade["exit_bar"] == 4
        assert result.equity.index.is_monotonic_increasing
        assert result.equity.notna().all()
        assert np.isfinite(result.equity.to_numpy()).all()

    def test_positive_mark_return_yields_cash(self, make_oi_market_data) -> None:
        data = make_oi_market_data(
            n_bars=8,
            mark_return_24h=[0.02] * 8,
            oi_change=[-1.0] * 8,
        )
        result = _run(data)
        assert len(result.trades) == 0
        assert np.allclose(result.equity.to_numpy(), 10_000.0)

    def test_rising_open_interest_yields_cash(self, make_oi_market_data) -> None:
        # FD-04: a rising OI value alone never produces a short.
        data = make_oi_market_data(
            n_bars=8,
            mark_return_24h=[-0.01] * 8,
            oi_change=[1.0] * 8,
        )
        result = _run(data)
        assert len(result.trades) == 0

    def test_missing_metrics_yields_no_signal(self, make_oi_market_data) -> None:
        # FD-03: a missing metric (NaN feature) is a no-signal interval, never
        # an imputed short.
        data = make_oi_market_data(
            n_bars=8,
            mark_return_24h=None,
            oi_change=[-1.0] * 8,
        )
        result = _run(data)
        assert len(result.trades) == 0

    def test_signal_delay_shifts_execution(self, make_oi_market_data) -> None:
        data = make_oi_market_data(
            n_bars=8,
            mark_return_24h=_FALLING,
            oi_change=[-1.0] * 8,
        )
        base = _run(data, delay=0)
        delayed = _run(data, delay=1)
        assert base.trades.iloc[0]["entry_bar"] == 1
        assert delayed.trades.iloc[0]["entry_bar"] == 2

    def test_costs_are_applied(self, make_oi_market_data) -> None:
        data = make_oi_market_data(
            n_bars=8,
            mark_return_24h=_FALLING,
            oi_change=[-1.0] * 8,
        )
        zero = _run(data, costs=ZERO_COSTS)
        costed = _run(data)
        assert costed.equity.iloc[-1] < zero.equity.iloc[-1]

    def test_short_receives_positive_funding(self, make_oi_market_data) -> None:
        # The short leg is credited positive funding while held into the
        # settlement bar.
        data = make_oi_market_data(
            n_bars=8,
            mark_return_24h=_FALLING,
            oi_change=[-1.0] * 8,
            funding={"2024-01-01 12:00": 0.001},
        )
        funded = _run(data, costs=ZERO_COSTS)
        unfunded = make_oi_market_data(
            n_bars=8,
            mark_return_24h=_FALLING,
            oi_change=[-1.0] * 8,
        )
        plain = _run(unfunded, costs=ZERO_COSTS)
        assert funded.equity.iloc[-1] > plain.equity.iloc[-1]
        assert funded.trades.iloc[0]["funding_pnl"] > 0.0

    def test_rejects_negative_delay(self, make_oi_market_data) -> None:
        data = make_oi_market_data()
        with pytest.raises(ValueError, match="signal_delay_bars"):
            run_open_interest_deleveraging_screen(data, CostModel(), signal_delay_bars=-1)


def test_screen_signature_is_frozen() -> None:
    from inspect import signature

    params = signature(run_open_interest_deleveraging_screen).parameters
    assert list(params) == ["market_data", "costs", "signal_delay_bars"]
    assert params["signal_delay_bars"].default == 1
