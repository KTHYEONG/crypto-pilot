from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.research.sleeve_blend.tournament as tournament_module
from src.research.sleeve_blend.contracts import PortfolioBlendTournamentRequest

_MODULE = "src.research.sleeve_blend.tournament"
_DISCOVERY_END = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")


def _good_equity() -> pd.Series:
    """Steady multi-year rise with a recoverable mid dip: passes every discovery gate."""
    idx = pd.date_range("2022-01-01", periods=3 * 2190, freq="4h", tz="UTC")
    growth = np.linspace(1.0, 2.0, len(idx))
    noise = 1.0 + 0.003 * np.sin(np.arange(len(idx)) / 18.0)
    eqv = 10_000.0 * growth * noise
    dip = np.ones(len(idx))
    dip[1500:1650] = np.linspace(1.0, 0.82, 150)
    dip[1650:1750] = np.linspace(0.82, 1.0, 100)
    return pd.Series(eqv * dip, index=idx, name="equity")


def _flat_equity() -> pd.Series:
    """Constant ledger: zero variance and a non-negative MDD -> infeasible screen."""
    idx = pd.date_range("2022-01-01", periods=3 * 2190, freq="4h", tz="UTC")
    return pd.Series(np.full(len(idx), 10_000.0), index=idx, name="equity")


def _concentrated_equity() -> pd.Series:
    """All net growth in one year: feasible but fails the dynamic-fold gate."""
    idx = pd.date_range("2022-01-01", periods=3 * 2190, freq="4h", tz="UTC")
    eqv = np.full(len(idx), 19_000.0)
    y1 = idx < pd.Timestamp("2023-01-01", tz="UTC")
    eqv[y1] = np.linspace(10_000.0, 19_000.0, int(y1.sum()))
    eqv[y1] = eqv[y1] * (
        1.0 - 0.10 * np.exp(-((np.arange(len(idx))[y1] - 800) / 60.0) ** 2)
    )
    return pd.Series(eqv, index=idx, name="equity")


def _trades(idx: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": ["A"] * 40,
        "entry_bar": np.arange(40),
        "exit_bar": np.arange(40) + 1,
        "entry_time": idx[:40],
        "exit_time": idx[1:41],
        "entry_price": [100.0] * 40,
        "exit_price": [104.0] * 40,
        "qty": [10.0] * 40,
        "reason": ["x"] * 40,
        "pnl": [40.0] * 40,
        "return_pct": [0.04] * 40,
        "funding_pnl": [0.0] * 40,
    })


_ALL_SOURCES = tournament_module.TOURNAMENT_RETURN_SOURCES


def _install(monkeypatch: pytest.MonkeyPatch, holder: dict[str, tuple[pd.Series, pd.DataFrame]]) -> None:
    """Route every source's backtest through pre-computed synthetic ledgers."""
    monkeypatch.setattr(
        f"{_MODULE}._load_universe_data",
        lambda universe, start, end: {
            symbol: (pd.DataFrame(), pd.Series(dtype="float64"))
            for symbol in universe.symbols
        },
    )
    monkeypatch.setattr(
        f"{_MODULE}._source_full_equity",
        lambda source, data, costs, delay: holder[source],
    )


def _request(*, discovery_end: pd.Timestamp = _DISCOVERY_END) -> PortfolioBlendTournamentRequest:
    return PortfolioBlendTournamentRequest(
        discovery_end=discovery_end, qualification_interval="365D",
        start=None, end=None,
    )


def test_pbgt_04_insufficient_discovery_history_stays_cash(monkeypatch) -> None:
    """PBGT-04: a discovery window before any data leaves every candidate CASH/REJECTED."""
    eq = _good_equity()
    holder = {
        source: (eq.copy(), _trades(eq.index)) for source in _ALL_SOURCES
    }
    _install(monkeypatch, holder)
    report = tournament_module.run_strategy_tournament(_request(
        discovery_end=pd.Timestamp("2021-06-30 23:59:59", tz="UTC"),
    ))
    assert report.selected_return_sources == ()
    assert report.blend_weights == ()
    assert all(not c.admitted for c in report.candidates)
    assert all(c.rejected_reason == "insufficient_data" for c in report.candidates)
    assert report.base_result.equity.nunique() == 1


def test_pbgt_04_failed_feasibility_stays_cash(monkeypatch) -> None:
    """PBGT-04: a candidate failing the feasibility screen is CASH/REJECTED."""
    flat = _flat_equity()
    holder = {source: (flat.copy(), pd.DataFrame()) for source in _ALL_SOURCES}
    _install(monkeypatch, holder)
    report = tournament_module.run_strategy_tournament(_request())
    assert report.selected_return_sources == ()
    for c in report.candidates:
        assert not c.admitted
        assert c.rejected_reason is not None
        assert c.rejected_reason.startswith("feasibility:")
    assert report.base_result.equity.nunique() == 1


def test_pbgt_04_failed_discovery_gate_cannot_enter_blend(monkeypatch) -> None:
    """PBGT-04: feasible but fold-failing discovery evidence stays CASH/REJECTED."""
    conc = _concentrated_equity()
    holder = {source: (conc.copy(), _trades(conc.index)) for source in _ALL_SOURCES}
    _install(monkeypatch, holder)
    report = tournament_module.run_strategy_tournament(_request())
    assert report.selected_return_sources == ()
    for c in report.candidates:
        assert not c.admitted
        assert c.discovery_observation is not None
        assert c.rejected_reason == "fold:gate_pass=False"


def test_pbgt_04_only_full_discovery_pass_gets_nonzero_weight(monkeypatch) -> None:
    """PBGT-04: only an independently gate-passing source enters the blend."""
    good = _good_equity()
    flat = _flat_equity()
    holder = {
        source: (good.copy(), _trades(good.index)) if i == 0 else (flat.copy(), pd.DataFrame())
        for i, source in enumerate(_ALL_SOURCES)
    }
    _install(monkeypatch, holder)
    report = tournament_module.run_strategy_tournament(_request())
    assert report.selected_return_sources == (tournament_module.DONCHIAN_LONG_ONLY,)
    assert report.blend_weights == (1.0,)
    admitted = next(c for c in report.candidates if c.admitted)
    assert admitted.rejected_reason is None
    assert report.base_result.equity.nunique() > 1


def test_pbgt_05_qualification_tail_cannot_alter_discovery_selection(monkeypatch) -> None:
    """PBGT-05: a superior qualification tail never changes selection, weights, or the schedule prefix."""
    base = _good_equity()
    holder = {source: (base.copy(), _trades(base.index)) for source in _ALL_SOURCES}
    _install(monkeypatch, holder)

    first = tournament_module.run_strategy_tournament(_request())

    tail = base.copy()
    post = tail.index > _DISCOVERY_END
    tail.loc[post] = tail.loc[post] * 2.0
    holder2 = {source: (tail.copy(), _trades(tail.index)) for source in _ALL_SOURCES}
    monkeypatch.setattr(
        f"{_MODULE}._source_full_equity",
        lambda source, data, costs, delay: holder2[source],
    )
    second = tournament_module.run_strategy_tournament(_request())

    assert second.selected_return_sources == first.selected_return_sources
    assert second.blend_weights == first.blend_weights
    prefix = first.leverage_schedule.index <= _DISCOVERY_END
    pd.testing.assert_series_equal(
        second.leverage_schedule[prefix], first.leverage_schedule[prefix],
    )
