from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.mhs.backtest.certification import DailyPortfolioEvidence
from src.mhs.execution import (
    ExecutionReplayWindow,
    FundingCoverageGap,
    SimulatedInventoryLedgerResult,
    StrategyExecutionReplayResult,
)
from src.mhs.frozen_research_candidate import FROZEN_MHS_TOP20_V1, FrozenMhsCandidate
from src.mhs.types import ExecutionSpec

import src.mhs.frozen_research_evidence as evidence_mod
from src.mhs.frozen_research_evidence import FrozenMhsReportPeriod, evaluate_frozen_mhs_research

_SYMBOLS = ("AAA", "BBB")
_DAY1 = pd.Timestamp("2021-06-01", tz="UTC")
_BASE_LIMITATIONS = (
    "CANDLE_FILLS_NO_ORDER_BOOK_DEPTH",
    "CANDLE_FILLS_NO_QUEUE_POSITION",
    "CANDLE_FILLS_NO_PARTICIPATION_CAPACITY",
    "DISCOVERY_YEARS_RESEARCH_ONLY_NO_FORWARD_CLAIM",
    "NO_DEPLOYMENT_VERDICT",
)


def _specs() -> tuple[ExecutionSpec, ExecutionSpec]:
    base = dataclasses.replace(ExecutionSpec(), taker_fee_bps=5.0, taker_slippage_bps=1.0)
    stress = dataclasses.replace(ExecutionSpec(), taker_fee_bps=5.0, taker_slippage_bps=13.0)
    assert base.one_way_taker_bps() == 6.0
    assert stress.one_way_taker_bps() == 18.0
    return base, stress


def _uncovered_periods() -> tuple[FrozenMhsReportPeriod, ...]:
    return (FrozenMhsReportPeriod(label="2022", start=pd.Timestamp("2022-01-01", tz="UTC"), end=pd.Timestamp("2022-12-31", tz="UTC")),)


def _candidate(labels: list[pd.Timestamp], aaa: float = 0.05, bbb: float = -0.05) -> FrozenMhsCandidate:
    weights = pd.DataFrame({"AAA": [aaa] * len(labels), "BBB": [bbb] * len(labels)}, index=pd.DatetimeIndex(labels, tz="UTC"), dtype="float64")
    avail = pd.DatetimeIndex([label - pd.Timedelta(hours=1) for label in labels], tz="UTC")
    return FrozenMhsCandidate(target_weights=weights, signal_available_at=avail, strategy=FROZEN_MHS_TOP20_V1)


def _frames(grid: pd.DatetimeIndex, price: float = 100.0) -> dict[str, pd.DataFrame]:
    cols = list(_SYMBOLS)
    return {
        "highs": pd.DataFrame(price + 1.0, index=grid, columns=cols, dtype="float64"),
        "lows": pd.DataFrame(price - 1.0, index=grid, columns=cols, dtype="float64"),
        "closes": pd.DataFrame(price, index=grid, columns=cols, dtype="float64"),
        "marks": pd.DataFrame(price, index=grid, columns=cols, dtype="float64"),
        "bar_funding": pd.DataFrame(0.0, index=grid, columns=cols, dtype="float64"),
        "quote_volumes": pd.DataFrame(1000.0, index=grid, columns=cols, dtype="float64"),
        "funding_known": pd.DataFrame(True, index=grid, columns=cols),
    }


def _engine_window(
    grid: pd.DatetimeIndex, candidate: FrozenMhsCandidate, labels: list[pd.Timestamp], **overrides: object
) -> ExecutionReplayWindow:
    weights = candidate.target_weights.loc[labels].copy()
    avail = pd.DatetimeIndex(
        [candidate.signal_available_at[candidate.target_weights.index.get_loc(label)] for label in labels], tz="UTC"
    )
    params: dict[str, object] = {
        "window_start": grid[0], "window_end": grid[-1], "columns": _SYMBOLS, "symbols": _SYMBOLS,
        "minute_grid": grid, "target_weights": weights, "signal_available_at": avail,
        "bar_available_at": grid + pd.Timedelta(minutes=3),
    }
    params.update(_frames(grid))
    params.update(overrides)
    return ExecutionReplayWindow(**params)  # type: ignore[arg-type]


def _engine_case() -> tuple[FrozenMhsCandidate, list[ExecutionReplayWindow]]:
    labels = [_DAY1 + pd.Timedelta(days=i) for i in (1, 2, 3)]
    candidate = _candidate(labels)
    first = _engine_window(pd.date_range(labels[0] - pd.Timedelta(hours=1), labels[1] + pd.Timedelta(hours=1), freq="3min", tz="UTC"), candidate, labels[:1])
    second = _engine_window(pd.date_range(labels[1] - pd.Timedelta(hours=2), labels[2] + pd.Timedelta(hours=2), freq="3min", tz="UTC"), candidate, labels[1:])
    return candidate, [first, second]


def test_configured_complete_periods_only(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate, base, stress = _hand_fixture()
    monkeypatch.setattr(evidence_mod, "replay_execution_window_batch", lambda *a, **k: (base, stress))
    complete = FrozenMhsReportPeriod(label="P1", start=pd.Timestamp("2021-06-02", tz="UTC"), end=pd.Timestamp("2021-06-03", tz="UTC"))
    incomplete = FrozenMhsReportPeriod(label="P9", start=pd.Timestamp("2021-06-05", tz="UTC"), end=pd.Timestamp("2021-06-10", tz="UTC"))
    base_spec, stress_spec = _specs()
    evidence = evaluate_frozen_mhs_research(
        candidate, [], initial_equity=100000.0, base_spec=base_spec, stress_spec=stress_spec,
        report_periods=(complete, incomplete),
    )
    assert float(evidence.period_metrics.loc["P1", "base_coverage"]) == 1.0
    assert bool(evidence.period_metrics.loc["P9", ["base_coverage", "stress_coverage"]].notna().all())
    assert bool(evidence.period_metrics.loc["P9"].drop(labels=["base_coverage", "stress_coverage"]).isna().all())
    assert "2022" not in evidence.period_metrics.index


def test_overlapping_reporting_periods_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate, base, stress = _hand_fixture()
    monkeypatch.setattr(evidence_mod, "replay_execution_window_batch", lambda *a, **k: (base, stress))
    overlapping = (
        FrozenMhsReportPeriod(label="A", start=pd.Timestamp("2021-06-02", tz="UTC"), end=pd.Timestamp("2021-06-03", tz="UTC")),
        FrozenMhsReportPeriod(label="B", start=pd.Timestamp("2021-06-03", tz="UTC"), end=pd.Timestamp("2021-06-04", tz="UTC")),
    )
    base_spec, stress_spec = _specs()
    with pytest.raises(DataIntegrityError, match="overlap"):
        evaluate_frozen_mhs_research(
            candidate, [], initial_equity=100000.0, base_spec=base_spec, stress_spec=stress_spec,
            report_periods=overlapping,
        )


def test_invalid_report_period_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate, base, stress = _hand_fixture()
    monkeypatch.setattr(evidence_mod, "replay_execution_window_batch", lambda *a, **k: (base, stress))
    base_spec, stress_spec = _specs()
    with pytest.raises(DataIntegrityError, match="non-empty tuple"):
        evaluate_frozen_mhs_research(
            candidate, [], initial_equity=100000.0, base_spec=base_spec, stress_spec=stress_spec,
            report_periods=(),
        )
    dup_labels = (
        FrozenMhsReportPeriod(label="A", start=pd.Timestamp("2021-06-02", tz="UTC"), end=pd.Timestamp("2021-06-03", tz="UTC")),
        FrozenMhsReportPeriod(label="A", start=pd.Timestamp("2021-06-05", tz="UTC"), end=pd.Timestamp("2021-06-06", tz="UTC")),
    )
    with pytest.raises(DataIntegrityError, match="unique"):
        evaluate_frozen_mhs_research(
            candidate, [], initial_equity=100000.0, base_spec=base_spec, stress_spec=stress_spec,
            report_periods=dup_labels,
        )
    with pytest.raises(DataIntegrityError, match="non-empty string"):
        FrozenMhsReportPeriod(label="", start=pd.Timestamp("2021-06-02", tz="UTC"), end=pd.Timestamp("2021-06-03", tz="UTC"))
    with pytest.raises(DataIntegrityError, match="valid timestamp"):
        FrozenMhsReportPeriod(label="X", start="2021-06-02", end=pd.Timestamp("2021-06-03", tz="UTC"))  # type: ignore[arg-type]
    with pytest.raises(DataIntegrityError, match="timezone-aware UTC"):
        FrozenMhsReportPeriod(
            label="X", start=pd.Timestamp("2021-06-02"), end=pd.Timestamp("2021-06-03"),
        )
    with pytest.raises(DataIntegrityError, match="start < end"):
        FrozenMhsReportPeriod(
            label="X", start=pd.Timestamp("2021-06-03", tz="UTC"), end=pd.Timestamp("2021-06-02", tz="UTC"),
        )


def test_evaluate_pairs_same_stream_costs_only() -> None:
    """Both bounds share windows and targets; only crossing cost changes."""
    candidate, windows = _engine_case()
    base_spec, stress_spec = _specs()
    evidence = evaluate_frozen_mhs_research(candidate, iter(windows), initial_equity=100000.0, base_spec=base_spec, stress_spec=stress_spec, report_periods=_uncovered_periods())
    assert evidence.base.fill_source == "OHLCV_IMMEDIATE_TAKER"
    assert evidence.stress.fill_source == "OHLCV_IMMEDIATE_TAKER"
    pd.testing.assert_series_equal(
        evidence.base.ledger.equity.index.to_series(), evidence.stress.ledger.equity.index.to_series()
    )
    assert list(evidence.base.simulated_fills["timestamp"]) == list(evidence.stress.simulated_fills["timestamp"])
    assert list(evidence.base.simulated_fills["symbol"]) == list(evidence.stress.simulated_fills["symbol"])
    assert set(evidence.base.simulated_fills["fee_bps"].unique()) == {6.0}
    assert set(evidence.stress.simulated_fills["fee_bps"].unique()) == {18.0}
    assert evidence.stress.all_intent_shortfall_bps > evidence.base.all_intent_shortfall_bps
    assert evidence.base_daily.returns.index.equals(evidence.stress_daily.returns.index)
    assert len(evidence.base_daily.returns) == 3
    assert evidence.limitations == _BASE_LIMITATIONS
    assert {f.name for f in dataclasses.fields(evidence)} == {
        "base", "stress", "base_daily", "stress_daily", "period_metrics", "limitations",
    }
    assert list(evidence.period_metrics.columns)[:4] == ["base_cagr", "stress_cagr", "base_max_drawdown", "stress_max_drawdown"]
    row2022 = evidence.period_metrics.loc["2022"]
    assert bool(row2022.drop(labels=["base_coverage", "stress_coverage"]).isna().all())
    assert float(row2022["base_coverage"]) == 0.0


def test_evaluate_rejects_cost_contract_and_capital() -> None:
    """An 8-bps base, a mechanically different stress, or bad capital emits nothing."""
    candidate, windows = _engine_case()
    base_spec, stress_spec = _specs()
    with pytest.raises(DataIntegrityError, match="6 bps"):
        evaluate_frozen_mhs_research(candidate, iter(windows), initial_equity=100000.0, base_spec=ExecutionSpec(), stress_spec=stress_spec, report_periods=_uncovered_periods())
    other = dataclasses.replace(stress_spec, passive_timeout_minutes=stress_spec.passive_timeout_minutes + 1)
    with pytest.raises(DataIntegrityError, match="except crossing cost"):
        evaluate_frozen_mhs_research(candidate, iter(windows), initial_equity=100000.0, base_spec=base_spec, stress_spec=other, report_periods=_uncovered_periods())
    for bad in (0.0, -10.0, float("nan")):
        with pytest.raises(DataIntegrityError, match="initial_equity"):
            evaluate_frozen_mhs_research(candidate, iter(windows), initial_equity=bad, base_spec=base_spec, stress_spec=stress_spec, report_periods=_uncovered_periods())


def test_evaluate_fails_closed_on_unknown_held_funding() -> None:
    """Unknown funding over held inventory cannot be scored as zero funding."""
    candidate, windows = _engine_case()
    cutoff = pd.Timestamp("2021-06-02 12:00", tz="UTC")
    unknown = []
    for window in windows:
        mask = window.minute_grid < cutoff
        known = pd.DataFrame(
            np.broadcast_to(np.asarray(mask)[:, None], (len(window.minute_grid), len(_SYMBOLS))),
            index=window.minute_grid, columns=list(_SYMBOLS),
        )
        unknown.append(dataclasses.replace(window, funding_known=known))
    base_spec, stress_spec = _specs()
    with pytest.raises(DataIntegrityError, match="unknown funding"):
        evaluate_frozen_mhs_research(candidate, iter(unknown), initial_equity=100000.0, base_spec=base_spec, stress_spec=stress_spec, report_periods=_uncovered_periods())


def test_evaluate_fails_closed_on_unpriced_exit() -> None:
    """A held symbol losing its marks with no settlement is never haircut-priced."""
    candidate, windows = _engine_case()
    blot: list[ExecutionReplayWindow] = []
    for window in windows:
        marks = window.marks.copy() if window.marks is not None else None
        if marks is not None:
            marks.loc[marks.index >= pd.Timestamp("2021-06-03", tz="UTC"), "AAA"] = float("nan")
        blot.append(dataclasses.replace(window, marks=marks))
    base_spec, stress_spec = _specs()
    with pytest.raises(DataIntegrityError, match="unpriced marks"):
        evaluate_frozen_mhs_research(candidate, iter(blot), initial_equity=100000.0, base_spec=base_spec, stress_spec=stress_spec, report_periods=_uncovered_periods())


def test_evaluate_uses_actual_funding_event_holdings() -> None:
    """Longs pay and shorts receive at the settlement bars they actually hold."""
    labels = [_DAY1 + pd.Timedelta(days=i) for i in (1, 2)]
    grid = pd.date_range(labels[0] - pd.Timedelta(hours=1), labels[1] + pd.Timedelta(hours=2), freq="3min", tz="UTC")
    long_candidate = _candidate(labels, aaa=0.10, bbb=0.0)
    frames = _frames(grid)
    frames["bar_funding"].loc[grid[(grid >= labels[0] + pd.Timedelta(hours=12)) & (grid < labels[0] + pd.Timedelta(hours=13))]] = 0.0001
    long_window = _engine_window(grid, long_candidate, labels, **frames)
    base_spec, stress_spec = _specs()
    long_evidence = evaluate_frozen_mhs_research(long_candidate, iter([long_window]), initial_equity=100000.0, base_spec=base_spec, stress_spec=stress_spec, report_periods=_uncovered_periods())
    charges = long_evidence.base.ledger.funding_charge
    funded = charges.loc[charges.index.to_series().between(labels[0] + pd.Timedelta(hours=12), labels[0] + pd.Timedelta(hours=13))]
    assert float(funded.sum()) > 0.0
    assert float(charges.drop(funded.index).abs().max()) == 0.0
    short_candidate = _candidate(labels, aaa=-0.10, bbb=0.0)
    short_window = _engine_window(grid, short_candidate, labels, **frames)
    short_evidence = evaluate_frozen_mhs_research(short_candidate, iter([short_window]), initial_equity=100000.0, base_spec=base_spec, stress_spec=stress_spec, report_periods=_uncovered_periods())
    short_charges = short_evidence.base.ledger.funding_charge
    assert float(short_charges.loc[funded.index].sum()) < 0.0


def test_evaluate_records_funding_gaps_as_limitation() -> None:
    """Recorded source gaps limit the claim without failing the valid replay."""
    candidate, windows = _engine_case()
    gap = FundingCoverageGap(symbol="BBB", start=windows[0].minute_grid[0], end=windows[0].minute_grid[-1], reason="OBSERVATION_GAP")
    gapped = [dataclasses.replace(windows[0], funding_coverage_gaps=(gap,)), windows[1]]
    base_spec, stress_spec = _specs()
    evidence = evaluate_frozen_mhs_research(candidate, iter(gapped), initial_equity=100000.0, base_spec=base_spec, stress_spec=stress_spec, report_periods=_uncovered_periods())
    assert evidence.limitations == (*_BASE_LIMITATIONS, "FUNDING_COVERAGE_GAPS_RECORDED")


def _hand_result(
    equity: pd.Series, funding: pd.Series, turnover: pd.Series, shortfall: float, forced: int = 0
) -> StrategyExecutionReplayResult:
    idx = equity.index
    zeros = pd.Series(np.zeros(len(idx)), index=idx, dtype="float64")
    ledger = SimulatedInventoryLedgerResult(
        equity=equity, net_returns=equity.pct_change().dropna(), simulated_units=None,
        mark_to_market_pnl=zeros, funding_charge=funding, fee_charge=zeros,
        fill_turnover=turnover, fill_source="OHLCV_IMMEDIATE_TAKER", mark_source="MARK_PRICE",
        primary_valid=True, invalid_reasons=(),
    )
    fills = pd.DataFrame({
        "timestamp": pd.Series([], dtype="datetime64[ns, UTC]"), "symbol": pd.Series([], dtype=object),
        "quantity_delta": pd.Series([], dtype="float64"), "fill_price": pd.Series([], dtype="float64"),
        "fee_bps": pd.Series([], dtype="float64"), "reason": pd.Series([], dtype=object),
        "pre_trade_equity": pd.Series([], dtype="float64"),
    })
    empty_units = pd.DataFrame(columns=list(_SYMBOLS))
    return StrategyExecutionReplayResult(
        simulated_fills=fills, ledger=ledger, simulated_units=empty_units,
        simulated_notional_weights=empty_units.copy(), fill_source="OHLCV_IMMEDIATE_TAKER",
        mark_source="MARK_PRICE", submit_times=pd.Series([], dtype="datetime64[ns, UTC]"),
        fill_times=pd.Series([], dtype="datetime64[ns, UTC]"), fill_count=0, unfilled_count=0,
        fallback_count=0, all_intent_shortfall_bps=shortfall, forced_exit_count=forced,
        forced_exit_notional=0.0, termination_counts={}, unsupported_assumptions=(),
        elapsed_seconds=0.0, data_gaps=(), funding_coverage_gaps=(),
        terminal_positions=(), ledger_available_at=pd.DatetimeIndex(idx, tz="UTC"),
    )


def _hand_fixture(trough: bool = False) -> tuple[FrozenMhsCandidate, StrategyExecutionReplayResult, StrategyExecutionReplayResult]:
    idx = pd.date_range("2021-06-01", "2021-06-04", freq="3min", tz="UTC")
    base_eq = pd.Series(100000.0, index=idx, dtype="float64")
    base_eq.loc[pd.Timestamp("2021-06-02 23:57", tz="UTC")] = 101000.0
    base_eq.loc[pd.Timestamp("2021-06-03 23:57", tz="UTC")] = 99000.0
    base_eq.loc[pd.Timestamp("2021-06-04 00:00", tz="UTC")] = 100500.0
    if trough:
        base_eq.loc[pd.Timestamp("2021-06-02 12:00", tz="UTC")] = 90000.0
    stress_eq = pd.Series(100000.0, index=idx, dtype="float64")
    stress_eq.loc[pd.Timestamp("2021-06-02 23:57", tz="UTC")] = 100500.0
    stress_eq.loc[pd.Timestamp("2021-06-03 23:57", tz="UTC")] = 99500.0
    stress_eq.loc[pd.Timestamp("2021-06-04 00:00", tz="UTC")] = 100250.0
    zeros = pd.Series(np.zeros(len(idx)), index=idx, dtype="float64")
    base_turnover = zeros.copy()
    base_turnover.loc[pd.Timestamp("2021-06-02 12:00", tz="UTC")] = 100.0
    base_turnover.loc[pd.Timestamp("2021-06-03 12:00", tz="UTC")] = 200.0
    base_turnover.loc[pd.Timestamp("2021-06-01 12:00", tz="UTC")] = 500.0
    base_funding = zeros.copy()
    base_funding.loc[pd.Timestamp("2021-06-02 06:00", tz="UTC")] = 5.0
    base_funding.loc[pd.Timestamp("2021-06-03 06:00", tz="UTC")] = -2.0
    base_funding.loc[pd.Timestamp("2021-06-01 06:00", tz="UTC")] = 999.0
    labels = [pd.Timestamp("2021-06-02", tz="UTC"), pd.Timestamp("2021-06-03", tz="UTC")]
    weights = pd.DataFrame({"AAA": [0.10, 0.30], "BBB": [-0.10, -0.10]}, index=pd.DatetimeIndex(labels, tz="UTC"), dtype="float64")
    candidate = FrozenMhsCandidate(
        target_weights=weights,
        signal_available_at=pd.DatetimeIndex([label - pd.Timedelta(hours=1) for label in labels], tz="UTC"),
        strategy=FROZEN_MHS_TOP20_V1,
    )
    base = _hand_result(base_eq, base_funding, base_turnover, 7.0)
    stress = _hand_result(stress_eq, zeros, zeros, 21.0, forced=1)
    return candidate, base, stress


def _covered_periods() -> tuple[FrozenMhsReportPeriod, ...]:
    return (FrozenMhsReportPeriod(label="P1", start=pd.Timestamp("2021-06-02", tz="UTC"), end=pd.Timestamp("2021-06-03", tz="UTC")),)


def test_evaluate_reports_covered_period_economics(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fully observed period reports exact paired economics and full coverage."""
    candidate, base, stress = _hand_fixture()
    monkeypatch.setattr(evidence_mod, "replay_execution_window_batch", lambda *a, **k: (base, stress))
    base_spec, stress_spec = _specs()
    evidence = evaluate_frozen_mhs_research(candidate, [], initial_equity=100000.0, base_spec=base_spec, stress_spec=stress_spec, report_periods=_covered_periods())
    row = evidence.period_metrics.loc["P1"]
    assert float(row["base_coverage"]) == 1.0
    assert float(row["base_cagr"]) == pytest.approx((1.01 * 99000.0 / 101000.0) ** (365.0 / 2.0) - 1.0)
    assert float(row["stress_cagr"]) == pytest.approx((1.005 * 99500.0 / 100500.0) ** (365.0 / 2.0) - 1.0)
    assert float(row["base_max_drawdown"]) == pytest.approx(1.0 - 99000.0 / 101000.0)
    assert float(row["base_annual_turnover"]) == pytest.approx(300.0 * 365.0 / 2.0)
    assert float(row["stress_annual_turnover"]) == pytest.approx(0.0)
    assert float(row["base_target_gross"]) == pytest.approx(0.3)
    assert float(row["base_funding_contribution"]) == pytest.approx(3.0 / 100000.0)
    assert float(row["base_fill_shortfall_bps"]) == pytest.approx(7.0)
    assert float(row["stress_fill_shortfall_bps"]) == pytest.approx(21.0)
    assert float(row["stress_forced_exits"]) == pytest.approx(1.0)


def test_evaluate_reports_marked_intraday_trough(monkeypatch: pytest.MonkeyPatch) -> None:
    """An intraday mark below adjacent closes deepens the reported drawdown."""
    candidate, base, stress = _hand_fixture(trough=True)
    monkeypatch.setattr(evidence_mod, "replay_execution_window_batch", lambda *a, **k: (base, stress))
    base_spec, stress_spec = _specs()
    evidence = evaluate_frozen_mhs_research(candidate, [], initial_equity=100000.0, base_spec=base_spec, stress_spec=stress_spec, report_periods=_covered_periods())
    reported = float(evidence.period_metrics.loc["P1", "base_max_drawdown"])
    assert reported == pytest.approx(0.10)
    assert reported > 1.0 - 99000.0 / 101000.0


def test_evaluate_rejects_mismatched_daily_intervals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Paired daily evidence on different intervals fails instead of blending."""
    candidate, base, stress = _hand_fixture()
    monkeypatch.setattr(evidence_mod, "replay_execution_window_batch", lambda *a, **k: (base, stress))
    calls = iter([0, 1])

    def _evidence(replay: StrategyExecutionReplayResult) -> DailyPortfolioEvidence:
        from src.mhs.backtest.certification import inventory_daily_evidence as _real

        good = _real(replay)
        if next(calls) == 0:
            return good
        shifted = good.returns.copy()
        shifted.index = shifted.index + pd.Timedelta(days=1)
        return dataclasses.replace(good, returns=shifted)

    monkeypatch.setattr(evidence_mod, "inventory_daily_evidence", _evidence)
    base_spec, stress_spec = _specs()
    with pytest.raises(DataIntegrityError, match="must match"):
        evaluate_frozen_mhs_research(candidate, [], initial_equity=100000.0, base_spec=base_spec, stress_spec=stress_spec, report_periods=_covered_periods())
