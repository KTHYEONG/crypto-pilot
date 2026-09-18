"""Invariant guards for coverage diagnostics on inventory replay reports."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.mhs.backtest.contracts import ProcessInventoryReport
from src.mhs.execution.contracts import (
    ExecutionReplayWindow,
    FundingAlignment,
    FundingCoverageGap,
    funding_coverage_gaps,
)
from src.mhs.reporting.inventory import export_inventory_json, persist_inventory_evidence
from src.mhs.types import ExecutionSpec
from src.mhs.execution.batch import replay_execution_windows


def _grid(periods: int = 6) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=periods, freq="3min", tz="UTC")


def _window(
    grid: pd.DatetimeIndex,
    weights: pd.DataFrame,
    known_values: list[bool],
    symbol: str = "AUSDT",
) -> ExecutionReplayWindow:
    px = pd.DataFrame({symbol: [100.0] * len(grid)}, index=grid, dtype="float64")
    known = pd.DataFrame({symbol: list(known_values)}, index=grid).astype(bool)
    rates = pd.DataFrame({symbol: [0.0] * len(grid)}, index=grid, dtype="float64")
    alignment = FundingAlignment(rates=rates, known=known, source_failures={})
    coverage = funding_coverage_gaps(alignment, grid)
    return ExecutionReplayWindow(
        window_start=grid[0],
        window_end=grid[-1],
        columns=(symbol,),
        symbols=(symbol,),
        minute_grid=grid,
        highs=px,
        lows=px,
        closes=px,
        marks=px,
        bar_funding=px * 0.0,
        target_weights=weights,
        signal_available_at=pd.DatetimeIndex([weights.index[0]]),
        quote_volumes=px * 0.0 + 1000.0,
        funding_known=known,
        bar_available_at=grid + pd.Timedelta(minutes=3),
        funding_coverage_gaps=coverage,
    )


def _spec() -> ExecutionSpec:
    return ExecutionSpec()


def test_inactive_uncovered_symbol_stays_diagnostic_only() -> None:
    grid = _grid()
    weights = pd.DataFrame({"AUSDT": [0.0]}, index=pd.DatetimeIndex([grid[0]]))
    window = _window(grid, weights, [False] * len(grid))
    result = replay_execution_windows((window,), 1000.0, "OHLCV_IMMEDIATE_TAKER", _spec())
    assert len(result.simulated_fills) == 0
    assert float(result.ledger.funding_charge.sum()) == 0.0
    assert len(result.funding_coverage_gaps) == 1
    assert result.ledger.primary_valid


def test_held_uncovered_symbol_invalidates_primary_evidence() -> None:
    grid = _grid(8)
    first = grid[:4]
    second = grid[3:]
    w1_weights = pd.DataFrame({"AUSDT": [0.5]}, index=pd.DatetimeIndex([first[0]]))
    w2_weights = pd.DataFrame({"AUSDT": [0.5]}, index=pd.DatetimeIndex([second[0]]))
    w1 = _window(first, w1_weights, [True] * len(first))
    w2 = _window(second, w2_weights, [False] * len(second))
    result = replay_execution_windows((w1, w2), 1000.0, "OHLCV_IMMEDIATE_TAKER", _spec())
    assert not result.ledger.primary_valid
    assert "MISSING_DATA" in set(result.ledger.invalid_reasons)
    assert any(g.code == "MISSING_HELD_FUNDING" for g in result.ledger.data_gaps)
    assert any(
        g.symbol == "AUSDT" for g in result.funding_coverage_gaps
    )
    assert any(g.start == second[0] for g in result.funding_coverage_gaps)


def test_active_order_uncovered_symbol_does_not_fill() -> None:
    grid = _grid()
    weights = pd.DataFrame({"AUSDT": [0.5]}, index=pd.DatetimeIndex([grid[0]]))
    window = _window(grid, weights, [False] * len(grid))
    result = replay_execution_windows((window,), 1000.0, "OHLCV_IMMEDIATE_TAKER", _spec())
    assert len(result.simulated_fills) == 0
    assert result.unfilled_count >= 1
    assert result.termination_counts.get("NO_FUNDING_UNFILLED", 0) >= 1
    assert len(result.funding_coverage_gaps) == 1


def test_report_serialization_preserves_intervals(tmp_path: Path) -> None:
    from tests.unit.mhs.test_inventory_persistence import _report

    gaps = (
        FundingCoverageGap(
            symbol="AUSDT",
            start=pd.Timestamp("2024-01-01T00:03:00", tz="UTC"),
            end=pd.Timestamp("2024-01-01T00:09:00", tz="UTC"),
            reason="OBSERVATION_GAP",
        ),
        FundingCoverageGap(
            symbol="BUSDT",
            start=pd.Timestamp("2024-01-01T00:00:00", tz="UTC"),
            end=pd.Timestamp("2024-01-01T00:06:00", tz="UTC"),
            reason="SOURCE_UNAVAILABLE",
        ),
    )
    report = _report()
    report = ProcessInventoryReport(
        proxy=report.proxy,
        base=report.base,
        stress=report.stress,
        gate=report.gate,
        resource_measurements=report.resource_measurements,
        memory_stats=report.memory_stats,
        funding_coverage_gaps=gaps,
    )
    summary_path, root = tmp_path / "summary.json", tmp_path / "evidence"
    out, _evid = persist_inventory_evidence(report, summary_path, evidence_root=root)
    exported = export_inventory_json(out, tmp_path / "full.json")
    payload = json.loads(exported.read_text(encoding="utf-8"))
    restored = payload["funding_coverage_gaps"]
    assert len(restored) == 2
    by_symbol = {entry["symbol"]: entry for entry in restored}
    assert by_symbol["AUSDT"]["reason"] == "OBSERVATION_GAP"
    assert by_symbol["AUSDT"]["start"] == gaps[0].start.isoformat()
    assert by_symbol["AUSDT"]["end"] == gaps[0].end.isoformat()
    assert by_symbol["BUSDT"]["reason"] == "SOURCE_UNAVAILABLE"
    assert by_symbol["BUSDT"]["start"] == gaps[1].start.isoformat()
    assert by_symbol["BUSDT"]["end"] == gaps[1].end.isoformat()
    assert isinstance(payload["base"]["funding_coverage_gaps"], list)
    assert isinstance(payload["stress"]["funding_coverage_gaps"], list)


def test_coverage_union_dedupes_identical_intervals() -> None:
    from src.mhs.backtest.inventory import _live_funding_coverage, _union_funding_coverage

    gap_a = FundingCoverageGap(
        symbol="AUSDT",
        start=pd.Timestamp("2024-01-01T00:03:00", tz="UTC"),
        end=pd.Timestamp("2024-01-01T00:09:00", tz="UTC"),
        reason="OBSERVATION_GAP",
    )
    gap_b = FundingCoverageGap(
        symbol="BUSDT",
        start=pd.Timestamp("2024-01-01T00:00:00", tz="UTC"),
        end=pd.Timestamp("2024-01-01T00:06:00", tz="UTC"),
        reason="SOURCE_UNAVAILABLE",
    )
    union = _union_funding_coverage((gap_a, gap_b), (gap_a,))
    assert union == tuple(sorted((gap_a, gap_b), key=lambda g: (g.start, g.end, g.symbol)))
    assert _union_funding_coverage((), ()) == ()
    assert _live_funding_coverage([]) == ()
    assert _live_funding_coverage([[None]]) == ()

    class _StubAcc:
        def __init__(self, gaps: dict[tuple[str, pd.Timestamp, pd.Timestamp, str], FundingCoverageGap]) -> None:
            self.funding_coverage_gaps = gaps

    stub = _StubAcc({(gap_a.symbol, gap_a.start, gap_a.end, gap_a.reason): gap_a})
    assert _live_funding_coverage([[stub]]) == (gap_a,)


def test_coverage_restore_failure_fails_closed(tmp_path: Path) -> None:
    from src.mhs.reporting.inventory import _restore_coverage

    from src.common.errors import DataIntegrityError

    with pytest.raises(DataIntegrityError):
        _restore_coverage(tmp_path, "coverage")
