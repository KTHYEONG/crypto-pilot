"""MHS evaluation core tests (second-level split by domain)."""

"""MHS evaluation core contract tests (everything not in a domain-specific split file)."""
"""Contract coverage for the MHS application evaluation resource telemetry."""
import numpy as np
import pandas as pd
import pytest
from src.mhs import evaluation as ev
from src.mhs.diagnostic_run import run_mhs_horizon_diagnostic
import src.mhs.resources as resources
from src.mhs.evaluation import (
    MhsDiagnosticRequest,
    _StageRecorder,
    _assert_cache_required_ledger_valid,
    _assert_execution_rss_budget,
)
from src.common.errors import DataIntegrityError
from src.mhs.types import ExecutionSpec
from src.mhs.execution import ExecutionReplayWindow, replay_execution_windows
from tests.unit.mhs.test_evaluation_appresearch import (  # noqa: F401
    _FOLD,
    _START,
    _assert_books_equal,
    _assert_regime_vol_mean_roster_masked,
    _build_book_outcome_args,
    _build_books_concurrent_args,
    _build_compact_report,
    _deployment_readiness,
    _dispatch_spec,
    _gap_mixed_replay,
    _passing_fold_report,
    _perf_opt_placebo_inputs,
    _pre_change_slow_book,
    _reference_bootstrap_ci,
    _reference_participation_warnings,
    _reference_placebo_percentile,
    _reference_resolve_ns_scalar,
    _reference_weights,
    _roster_mask_panel_inputs,
    _sequential_book_reports,
    _signal_disagreement_panel,
    _slow_book_panel_inputs,
    _synthetic_ledger,
    _write_3m_cache,
    _write_quote_volume_market,
)

def test_mhs_resource_measurement_records_ordered_stage_data() -> None:
    recorder = _StageRecorder(log_run=False)
    recorder.record("unit_stage", grid_bars=3, n_symbols=2, fill_count=1)

    records = recorder.records
    assert len(records) == 1
    record = records[0]
    assert record.stage == "unit_stage"
    assert record.elapsed_ms >= 0
    assert record.rss_bytes > 0
    assert record.peak_rss_bytes == record.rss_bytes
    assert record.window_start is None
    assert record.window_end is None
    assert record.active_symbols is None
    assert record.grid_bars == 3
    assert record.n_symbols == 2
    assert record.fill_count == 1

def test_mhs_mem_03_rss_budget_fails_closed(monkeypatch) -> None:
    """MHS-MEM-03: a configured RSS budget produces deterministic
    DataIntegrityError provenance rather than a process-level OOM or a valid
    partial report."""
    assert MhsDiagnosticRequest().max_rss_bytes is None
    with pytest.raises(ValueError, match="max_rss_bytes"):
        MhsDiagnosticRequest(max_rss_bytes=0)
    with pytest.raises(ValueError, match="max_rss_bytes"):
        MhsDiagnosticRequest(max_rss_bytes=-1)
    assert MhsDiagnosticRequest(max_rss_bytes=1_000_000_000).max_rss_bytes == 1_000_000_000

    monkeypatch.setattr("src.mhs.resources._current_rss_bytes", lambda: 5_000_000_000)
    with pytest.raises(DataIntegrityError, match="execution RSS budget exceeded") as excinfo:
        _assert_execution_rss_budget("execution_window", 1_000_000_000, 7)
    message = str(excinfo.value)
    assert "stage=execution_window" in message
    assert "observed_rss=5000000000" in message
    assert "budget=1000000000" in message
    assert "completed_windows=7" in message
    _assert_execution_rss_budget("execution_window", None, 7)

def test_mhs_mem_04_strict_gap_preserved() -> None:
    """MHS-MEM-04: cache_required continues to fail closed on MISSING_HELD_MARK
    for a held-mark fixture; stale carry remains explicit diagnostic mode."""
    grid = pd.date_range("2021-01-01", periods=48, freq="5min", tz="UTC")
    px = pd.DataFrame({"A": [100.0] * len(grid)}, index=grid)
    marks = px.copy()
    marks.loc[grid[20]:grid[25], "A"] = np.nan
    target = pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("2021-01-01 00:00", tz="UTC")])
    signals = pd.DatetimeIndex([pd.Timestamp("2021-01-01 01:00", tz="UTC")])
    window = ExecutionReplayWindow(
        window_start=grid[0],
        window_end=grid[-1],
        columns=("A",),
        symbols=("A",),
        minute_grid=grid,
        highs=px,
        lows=px,
        closes=px,
        marks=marks,
        bar_funding=pd.DataFrame(0.0, index=grid, columns=["A"]),
        target_weights=target,
        signal_available_at=signals,
    )
    replay = replay_execution_windows(
        [window], 1.0, "OHLCV_STRICT_PROXY", ExecutionSpec(),
    )
    assert replay.event_snapshots_retained is False
    gap_codes = {g.code for g in replay.data_gaps}
    assert "MISSING_HELD_MARK" in gap_codes
    assert replay.ledger.primary_valid is False
    with pytest.raises(DataIntegrityError, match="invalid"):
        _assert_cache_required_ledger_valid("held_mark_book", replay)

    assert replay.ledger.primary_valid is False

def test_ram_guard_resolve_budget(monkeypatch) -> None:
    # SCENARIO_MHS_RAM_GUARD_RESOLVE_BUDGET: _resolve_ram_budget maps the
    # request into (budget_bytes, reserve_bytes). ram_guard=False disables the
    # guard; ram_guard=True auto-derives 85% of total RAM and the reserve floor
    # max(5% of total, 256 MiB); an explicit max_rss_bytes overrides the budget
    # fraction; psutil failure / non-positive total yields (None, None).
    from src.mhs.types import (
        RAM_BUDGET_FRACTION,
        RAM_RESERVE_FLOOR_BYTES,
        RAM_RESERVE_FRACTION,
    )

    class _FakeMem:
        total: int
        available: int
        def __init__(self, total: int, available: int) -> None:
            self.total = total
            self.available = available

    assert ev._resolve_ram_budget(None, False) == (None, None)

    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: _FakeMem(8 * 2**30, 4 * 2**30))
    budget, reserve = ev._resolve_ram_budget(None, True)
    assert budget == int(8 * 2**30 * RAM_BUDGET_FRACTION)
    assert reserve == max(int(8 * 2**30 * RAM_RESERVE_FRACTION), RAM_RESERVE_FLOOR_BYTES)

    explicit, reserve2 = ev._resolve_ram_budget(123456789, True)
    assert explicit == 123456789
    assert reserve2 == reserve

    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: _FakeMem(0, 0))
    assert ev._resolve_ram_budget(None, True) == (None, None)

    def _boom() -> _FakeMem:
        raise RuntimeError("psutil unavailable")
    monkeypatch.setattr(resources.psutil, "virtual_memory", _boom)
    assert ev._resolve_ram_budget(None, True) == (None, None)

def test_ram_guard_stage_barrier_fails_closed(monkeypatch) -> None:
    # SCENARIO_MHS_RAM_GUARD_STAGE_BARRIER_FAIL_CLOSED: _assert_stage_rss_budget
    # fails closed deterministically -- process RSS above the budget raises a
    # DataIntegrityError naming the stage; system available memory below the
    # reserve raises; (None, None) is a no-op.
    with pytest.raises(ev.DataIntegrityError, match="RAM budget exceeded at stage 'test_stage'"):
        ev._assert_stage_rss_budget("test_stage", 1, None)

    class _FakeMem:
        total: int
        available: int
        def __init__(self, total: int, available: int) -> None:
            self.total = total
            self.available = available

    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: _FakeMem(8 * 2**30, 100))
    with pytest.raises(ev.DataIntegrityError, match="reserve breached at stage 'test_reserve'"):
        ev._assert_stage_rss_budget("test_reserve", None, 4096)

    ev._assert_stage_rss_budget("noop", None, None)

def test_ram_guard_request_field() -> None:
    # SCENARIO_MHS_RAM_GUARD_REQUEST_FIELD: ram_guard defaults True on the
    # request; a non-bool value fails closed; max_rss_bytes stays None (auto
    # resolution happens at run time).
    assert MhsDiagnosticRequest().ram_guard is True
    assert MhsDiagnosticRequest().max_rss_bytes is None
    with pytest.raises(ValueError, match="ram_guard"):
        MhsDiagnosticRequest(ram_guard="yes")
    assert MhsDiagnosticRequest(ram_guard=False).ram_guard is False

@pytest.mark.slow
def test_pipeline_ram_guard_fails_closed_before_oom(mhs_market_long) -> None:
    # SCENARIO_MHS_PIPELINE_RAM_GUARD_FAILS_CLOSED_BEFORE_OOM: a tiny explicit
    # budget makes run_mhs_horizon_diagnostic fail closed with a serializable
    # terminal COMPLETE report (MHS-28) carrying RESOURCE_BUDGET_BREACH instead
    # of letting the OS OOM killer terminate the process or raising uncaught.
    root, end = mhs_market_long
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8, max_rss_bytes=1,
    )
    report = run_mhs_horizon_diagnostic(request)
    assert report.status == "COMPLETE"
    assert ev.GO_REASON_RESOURCE_BREACH in report.research_go.reason_codes
    assert report.research_go.eligible is False
    for book in report.books.values():
        assert book.failure is not None
        assert book.failure.reason == ev.GO_REASON_RESOURCE_BREACH
    assert report.resource_measurements, "terminal report must retain stage telemetry"
