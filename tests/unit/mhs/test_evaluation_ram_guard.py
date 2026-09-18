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

    monkeypatch.setattr("src.mhs.resources._current_tree_pss_bytes", lambda: 5_000_000_000)
    with pytest.raises(DataIntegrityError, match="execution RSS budget exceeded") as excinfo:
        _assert_execution_rss_budget("execution_window", 1_000_000_000, 7)
    message = str(excinfo.value)
    assert "stage=execution_window" in message
    assert "observed_tree_pss=5000000000" in message
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
    # Terminal-only disclosure (ledger_terminal_only): a held gap with no
    # resuming fill stays open and invalid without crashing the book here;
    # deployment stays blocked downstream by backtest-reliability
    # certification instead of failing the replay.
    from src.mhs.evaluation.integrity import ledger_terminal_only
    assert ledger_terminal_only(replay.data_gaps, replay.simulated_fills) is True
    assert replay.ledger.primary_valid is False

    assert replay.ledger.primary_valid is False

def test_ram_guard_resolve_budget(monkeypatch) -> None:
    # SCENARIO_MHS_RAM_GUARD_RESOLVE_BUDGET: _resolve_ram_budget maps the
    # request into (budget_bytes, reserve_bytes). ram_guard=False disables the
    # guard; ram_guard=True auto-derives from the smaller host/cgroup limit and
    # the reserve floor max(5% of total, 256 MiB, 2 GiB); an explicit
    # max_rss_bytes overrides the budget fraction under the 2.5 GiB adoption
    # ceiling; psutil failure / non-positive total fails closed.
    from src.mhs.types import (
        RAM_BUDGET_FRACTION,
        RAM_RESERVE_FLOOR_BYTES,
        RAM_RESERVE_FRACTION,
    )
    from src.mhs.resources import MHS_AVAILABLE_FLOOR_BYTES, MHS_TREE_PSS_BUDGET_BYTES

    class _FakeMem:
        total: int
        available: int
        def __init__(self, total: int, available: int) -> None:
            self.total = total
            self.available = available

    assert ev._resolve_ram_budget(None, False) == (None, None)

    monkeypatch.setattr(resources, "_read_cgroup_limit_bytes", lambda: None)
    monkeypatch.setattr(resources, "_read_cgroup_remaining_bytes", lambda: None)
    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: _FakeMem(8 * 2**30, 4 * 2**30))
    budget, reserve = ev._resolve_ram_budget(None, True)
    assert budget == min(int(8 * 2**30 * RAM_BUDGET_FRACTION), MHS_TREE_PSS_BUDGET_BYTES)
    assert reserve == max(int(8 * 2**30 * RAM_RESERVE_FRACTION), RAM_RESERVE_FLOOR_BYTES, MHS_AVAILABLE_FLOOR_BYTES)

    explicit, reserve2 = ev._resolve_ram_budget(123456789, True)
    assert explicit == 123456789
    assert reserve2 == reserve

    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: _FakeMem(0, 0))
    with pytest.raises(DataIntegrityError, match="telemetry unavailable"):
        ev._resolve_ram_budget(None, True)

    def _boom() -> _FakeMem:
        raise RuntimeError("psutil unavailable")
    monkeypatch.setattr(resources.psutil, "virtual_memory", _boom)
    with pytest.raises(DataIntegrityError, match="telemetry unavailable"):
        ev._resolve_ram_budget(None, True)

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


_GIB = 2**30


def test_resolve_mhs_memory_budget_conservative_defaults(monkeypatch) -> None:
    """Conservative defaults: adequate capacity keeps 2.5/1.5 GiB ceilings and 2 GiB reserve."""
    from src.mhs.resources import (
        MHS_AVAILABLE_FLOOR_BYTES,
        MHS_REPLAY_BUDGET_BYTES,
        MHS_TREE_PSS_BUDGET_BYTES,
        resolve_mhs_memory_budget,
    )

    monkeypatch.setattr(resources, "_host_total_bytes", lambda: 16 * _GIB)
    monkeypatch.setattr(resources, "_read_cgroup_limit_bytes", lambda: None)
    resolved = resolve_mhs_memory_budget(None)
    assert resolved.total_tree_pss_bytes == MHS_TREE_PSS_BUDGET_BYTES
    assert resolved.replay_tree_pss_bytes == MHS_REPLAY_BUDGET_BYTES
    assert resolved.min_available_bytes == MHS_AVAILABLE_FLOOR_BYTES


def test_resolve_mhs_memory_budget_preserves_explicit_desktop_ceiling(monkeypatch) -> None:
    """Explicit desktop ceiling: 4/3 GiB on 16 GiB capacity is preserved without historical clipping."""
    from src.mhs.resources import MhsMemoryBudget, resolve_mhs_memory_budget

    explicit = MhsMemoryBudget(
        total_tree_pss_bytes=4 * _GIB, replay_tree_pss_bytes=3 * _GIB, min_available_bytes=2 * _GIB,
    )
    monkeypatch.setattr(resources, "_host_total_bytes", lambda: 16 * _GIB)
    monkeypatch.setattr(resources, "_read_cgroup_limit_bytes", lambda: 16 * _GIB)
    resolved = resolve_mhs_memory_budget(explicit)
    assert resolved.total_tree_pss_bytes == 4 * _GIB
    assert resolved.replay_tree_pss_bytes == 3 * _GIB
    assert resolved.min_available_bytes == 2 * _GIB


def test_resolve_mhs_memory_budget_clamps_to_physical_capacity(monkeypatch) -> None:
    """Physical capacity clamp: 4 GiB effective capacity bounds both ceilings to 2 GiB."""
    from src.mhs.resources import MhsMemoryBudget, resolve_mhs_memory_budget

    requested = MhsMemoryBudget(
        total_tree_pss_bytes=4 * _GIB, replay_tree_pss_bytes=3 * _GIB, min_available_bytes=2 * _GIB,
    )
    monkeypatch.setattr(resources, "_host_total_bytes", lambda: 4 * _GIB)
    monkeypatch.setattr(resources, "_read_cgroup_limit_bytes", lambda: None)
    resolved = resolve_mhs_memory_budget(requested)
    assert resolved.total_tree_pss_bytes == 2 * _GIB
    assert resolved.replay_tree_pss_bytes == 2 * _GIB
    assert resolved.min_available_bytes == 2 * _GIB


def test_resolve_mhs_memory_budget_rejects_impossible_reserve(monkeypatch) -> None:
    """Impossible reserve: finite capacity at or below the requested reserve raises a typed rejection."""
    from src.mhs.resources import MhsMemoryBudget, MhsResourceAdmissionError, resolve_mhs_memory_budget

    requested = MhsMemoryBudget(
        total_tree_pss_bytes=4 * _GIB, replay_tree_pss_bytes=3 * _GIB, min_available_bytes=2 * _GIB,
    )
    monkeypatch.setattr(resources, "_host_total_bytes", lambda: 2 * _GIB)
    monkeypatch.setattr(resources, "_read_cgroup_limit_bytes", lambda: None)
    with pytest.raises(MhsResourceAdmissionError) as excinfo:
        resolve_mhs_memory_budget(requested)
    assert excinfo.value.error_code == "MEMORY_RESERVE"
    with pytest.raises(ValueError, match="budget"):
        resolve_mhs_memory_budget("not-a-budget")

    def _boom() -> int:
        raise DataIntegrityError("no host telemetry")

    monkeypatch.setattr(resources, "_host_total_bytes", _boom)
    with pytest.raises(MhsResourceAdmissionError) as excinfo:
        resolve_mhs_memory_budget(None)
    assert excinfo.value.error_code == "RESOURCE_TELEMETRY"
    assert isinstance(excinfo.value.__cause__, DataIntegrityError)


def test_assert_mhs_allocation_budget_rejects_resident_ceiling(monkeypatch) -> None:
    """Resident allocation ceiling: PSS plus estimate above the ceiling raises typed MEMORY_BUDGET."""
    from src.mhs.resources import MhsResourceAdmissionError, assert_mhs_allocation_budget

    monkeypatch.setattr(resources, "_current_tree_pss_bytes", lambda: 3 * _GIB)
    monkeypatch.setattr(resources, "_tree_headroom_bytes", lambda: 8 * _GIB)
    monkeypatch.setattr(resources, "_current_tree_swap_bytes", lambda: None)
    with pytest.raises(MhsResourceAdmissionError) as excinfo:
        assert_mhs_allocation_budget(
            estimated_bytes=_GIB, budget_bytes=3 * _GIB, reserve_bytes=_GIB,
            stage="execution_allocation", initial_swap_bytes=None,
        )
    assert excinfo.value.error_code == "MEMORY_BUDGET"
    assert excinfo.value.stage == "execution_allocation"
    assert "tree_pss=" in str(excinfo.value)
    assert "budget=" in str(excinfo.value)
    assert_mhs_allocation_budget(
        estimated_bytes=0, budget_bytes=None, reserve_bytes=_GIB,
        stage="execution_allocation", initial_swap_bytes=None,
    )


def test_assert_mhs_allocation_budget_rejects_exhausted_cgroup(monkeypatch) -> None:
    """Cgroup exhausted: zero finite cgroup remaining rejects via MEMORY_RESERVE despite host availability."""
    from src.mhs.resources import MhsResourceAdmissionError, assert_mhs_allocation_budget

    class _FakeMem:
        total = 16 * _GIB
        available = 12 * _GIB

    monkeypatch.setattr(resources, "_current_tree_pss_bytes", lambda: _GIB)
    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: _FakeMem())
    monkeypatch.setattr(resources, "_read_cgroup_remaining_bytes", lambda: 0)
    monkeypatch.setattr(resources, "_current_tree_swap_bytes", lambda: None)
    with pytest.raises(MhsResourceAdmissionError) as excinfo:
        assert_mhs_allocation_budget(
            estimated_bytes=0, budget_bytes=8 * _GIB, reserve_bytes=2 * _GIB,
            stage="execution_allocation", initial_swap_bytes=None,
        )
    assert excinfo.value.error_code == "MEMORY_RESERVE"


def test_window_barrier_governed_by_pss_not_parent_rss(monkeypatch) -> None:
    """Shared pages: parent RSS above the ceiling with tree PSS inside admits the window."""
    from src.mhs.resources import MhsResourceAdmissionError, _assert_execution_rss_budget

    monkeypatch.setattr(resources, "_current_tree_pss_bytes", lambda: _GIB)
    monkeypatch.setattr(resources, "_current_rss_bytes", lambda: 8 * _GIB)
    monkeypatch.setattr(resources, "_tree_headroom_bytes", lambda: 8 * _GIB)
    _assert_execution_rss_budget("process_3m_window_0", 2 * _GIB, 0, reserve_bytes=_GIB)
    with pytest.raises(MhsResourceAdmissionError) as excinfo:
        _assert_execution_rss_budget("process_3m_window_0", None, 0, reserve_bytes=16 * _GIB)
    assert excinfo.value.error_code == "MEMORY_RESERVE"


def test_assert_mhs_allocation_budget_compares_swap_against_baseline(monkeypatch) -> None:
    """Swap baseline: unchanged swap passes, growth raises SWAP_GROWTH, missing baseline stays null."""
    from src.mhs.resources import MhsResourceAdmissionError, assert_mhs_allocation_budget

    monkeypatch.setattr(resources, "_current_tree_pss_bytes", lambda: _GIB)
    monkeypatch.setattr(resources, "_tree_headroom_bytes", lambda: 8 * _GIB)
    monkeypatch.setattr(resources, "_current_tree_swap_bytes", lambda: 4096)
    assert_mhs_allocation_budget(
        estimated_bytes=0, budget_bytes=8 * _GIB, reserve_bytes=_GIB,
        stage="execution_allocation", initial_swap_bytes=4096,
    )
    monkeypatch.setattr(resources, "_current_tree_swap_bytes", lambda: 8192)
    with pytest.raises(MhsResourceAdmissionError) as excinfo:
        assert_mhs_allocation_budget(
            estimated_bytes=0, budget_bytes=8 * _GIB, reserve_bytes=_GIB,
            stage="execution_allocation", initial_swap_bytes=4096,
        )
    assert excinfo.value.error_code == "SWAP_GROWTH"
    monkeypatch.setattr(resources, "_current_tree_swap_bytes", lambda: None)
    assert_mhs_allocation_budget(
        estimated_bytes=0, budget_bytes=8 * _GIB, reserve_bytes=_GIB,
        stage="execution_allocation", initial_swap_bytes=4096,
    )
    assert_mhs_allocation_budget(
        estimated_bytes=0, budget_bytes=8 * _GIB, reserve_bytes=None,
        stage="execution_allocation", initial_swap_bytes=None,
    )


def test_assert_mhs_allocation_budget_fails_closed_on_unreadable_telemetry(monkeypatch) -> None:
    """Unreadable telemetry: missing PSS or host headroom raises RESOURCE_TELEMETRY with chained cause."""
    from src.mhs.resources import MhsResourceAdmissionError, assert_mhs_allocation_budget

    def _boom_pss() -> int:
        raise DataIntegrityError("cannot enumerate process tree")

    monkeypatch.setattr(resources, "_current_tree_pss_bytes", _boom_pss)
    with pytest.raises(MhsResourceAdmissionError) as excinfo:
        assert_mhs_allocation_budget(
            estimated_bytes=0, budget_bytes=8 * _GIB, reserve_bytes=_GIB,
            stage="execution_allocation", initial_swap_bytes=None,
        )
    assert excinfo.value.error_code == "RESOURCE_TELEMETRY"
    assert isinstance(excinfo.value.__cause__, DataIntegrityError)
    monkeypatch.setattr(resources, "_current_tree_pss_bytes", lambda: _GIB)

    def _boom_headroom() -> int:
        raise DataIntegrityError("cannot read available memory")

    monkeypatch.setattr(resources, "_tree_headroom_bytes", _boom_headroom)
    with pytest.raises(MhsResourceAdmissionError) as excinfo:
        assert_mhs_allocation_budget(
            estimated_bytes=0, budget_bytes=8 * _GIB, reserve_bytes=_GIB,
            stage="execution_allocation", initial_swap_bytes=None,
        )
    assert excinfo.value.error_code == "RESOURCE_TELEMETRY"
    assert isinstance(excinfo.value.__cause__, DataIntegrityError)


def test_plan_mhs_execution_bars_rejects_impossible_minimum(monkeypatch) -> None:
    """Minimum legal piece: an unfittable timeout-preserving minimum raises a typed rejection."""
    from src.mhs.resources import MhsExecutionAllocation, MhsResourceAdmissionError, plan_mhs_execution_bars

    monkeypatch.setattr(resources, "_current_tree_pss_bytes", lambda: 0)
    monkeypatch.setattr(resources, "_tree_headroom_bytes", lambda: 8 * _GIB)
    allocation = MhsExecutionAllocation(fixed_bytes=0, bytes_per_bar=_GIB, decoder_bytes=0)
    with pytest.raises(MhsResourceAdmissionError) as excinfo:
        plan_mhs_execution_bars(
            requested_bars=10, minimum_bars=2, allocation=allocation,
            budget_bytes=_GIB, reserve_bytes=_GIB,
        )
    assert excinfo.value.error_code == "MEMORY_BUDGET"
    with pytest.raises(MhsResourceAdmissionError) as excinfo:
        plan_mhs_execution_bars(
            requested_bars=10, minimum_bars=2, allocation=allocation,
            budget_bytes=64 * _GIB, reserve_bytes=8 * _GIB,
        )
    assert excinfo.value.error_code == "MEMORY_RESERVE"
    assert plan_mhs_execution_bars(
        requested_bars=4, minimum_bars=2, allocation=allocation,
        budget_bytes=8 * _GIB, reserve_bytes=_GIB,
    ) == 4


def test_assert_mhs_allocation_budget_rejects_invalid_contract() -> None:
    """Allocation validation: invalid sizes, limits or stages raise ValueError; dual-None stays a no-op."""
    from src.mhs.resources import assert_mhs_allocation_budget

    assert assert_mhs_allocation_budget(estimated_bytes=-1, budget_bytes=None, reserve_bytes=None) is None
    with pytest.raises(ValueError, match="estimated_bytes"):
        assert_mhs_allocation_budget(estimated_bytes=-1, budget_bytes=1, reserve_bytes=None)
    with pytest.raises(ValueError, match="stage"):
        assert_mhs_allocation_budget(estimated_bytes=0, budget_bytes=1, reserve_bytes=None, stage="")
    with pytest.raises(ValueError, match="budget_bytes"):
        assert_mhs_allocation_budget(estimated_bytes=0, budget_bytes=0, reserve_bytes=None)
    with pytest.raises(ValueError, match="reserve_bytes"):
        assert_mhs_allocation_budget(estimated_bytes=0, budget_bytes=None, reserve_bytes=-5)
    with pytest.raises(ValueError, match="initial_swap_bytes"):
        assert_mhs_allocation_budget(
            estimated_bytes=0, budget_bytes=1, reserve_bytes=None, initial_swap_bytes=-1,
        )


def test_current_mhs_headroom_bytes_reports_effective_headroom(monkeypatch) -> None:
    """Headroom observation: minimum of host available and known cgroup remainder."""
    from src.mhs.resources import current_mhs_headroom_bytes

    class _FakeMem:
        total = 16 * _GIB
        available = 12 * _GIB

    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: _FakeMem())
    monkeypatch.setattr(resources, "_read_cgroup_remaining_bytes", lambda: 3 * _GIB)
    assert current_mhs_headroom_bytes() == 3 * _GIB
    monkeypatch.setattr(resources, "_read_cgroup_remaining_bytes", lambda: None)
    assert current_mhs_headroom_bytes() == 12 * _GIB


def test_stage_barriers_fail_closed_on_unreadable_telemetry(monkeypatch) -> None:
    """Barrier telemetry: unreadable PSS or headroom raises typed RESOURCE_TELEMETRY."""
    from src.mhs.resources import (
        MhsResourceAdmissionError,
        _assert_execution_rss_budget,
        _assert_stage_rss_budget,
    )

    def _boom_pss() -> int:
        raise DataIntegrityError("unreadable live process")

    monkeypatch.setattr(resources, "_current_tree_pss_bytes", _boom_pss)
    with pytest.raises(MhsResourceAdmissionError) as excinfo:
        _assert_stage_rss_budget("test_stage", 1, None)
    assert excinfo.value.error_code == "RESOURCE_TELEMETRY"
    with pytest.raises(MhsResourceAdmissionError) as excinfo:
        _assert_execution_rss_budget("execution_window", 1, 0)
    assert excinfo.value.error_code == "RESOURCE_TELEMETRY"
    monkeypatch.setattr(resources, "_current_tree_pss_bytes", lambda: 0)

    def _boom_headroom() -> int:
        raise DataIntegrityError("cannot read available memory")

    monkeypatch.setattr(resources, "_tree_headroom_bytes", _boom_headroom)
    with pytest.raises(MhsResourceAdmissionError) as excinfo:
        _assert_stage_rss_budget("test_reserve", None, 4096)
    assert excinfo.value.error_code == "RESOURCE_TELEMETRY"
    with pytest.raises(MhsResourceAdmissionError) as excinfo:
        _assert_execution_rss_budget("execution_window", None, 0, reserve_bytes=4096)
    assert excinfo.value.error_code == "RESOURCE_TELEMETRY"
