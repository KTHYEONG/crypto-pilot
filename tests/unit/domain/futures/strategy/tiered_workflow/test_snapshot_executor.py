from __future__ import annotations

from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.domain.futures.strategy.candidate_contracts import Layer1EvidenceSnapshot
from src.domain.futures.strategy.tiered_workflow.snapshot_executor import (
    L1SnapshotExecutionReport,
    L1SnapshotTask,
    _clear_globals,
    _execute_snapshot_task,
    _snapshot_matured_count,
    execute_l1_snapshot_batch,
)


def _make_snapshot(as_of_idx: int) -> Layer1EvidenceSnapshot:
    return Layer1EvidenceSnapshot(
        as_of_idx=as_of_idx,
        evidence=(),
        registry=MagicMock(),
        matured_event_count=0,
    )


def _fake_execute_snapshot_task(task: L1SnapshotTask) -> Layer1EvidenceSnapshot:
    return _make_snapshot(task.as_of_idx)


@pytest.fixture(autouse=True)
def _reset_globals():
    import src.domain.futures.strategy.tiered_workflow.snapshot_executor as se

    se._GLOBAL_EVIDENCE_STORE = None
    se._GLOBAL_CFG = None
    se._GLOBAL_SYMBOLS = None
    se._GLOBAL_SEED = 0
    se._GLOBAL_PROBE_DIVERSITY_CORR = None


def test_snapshot_task_dataclass() -> None:
    task = L1SnapshotTask(snapshot_offset=0, as_of_idx=100)
    assert task.snapshot_offset == 0
    assert task.as_of_idx == 100


def test_snapshot_execution_report_dataclass() -> None:
    report = L1SnapshotExecutionReport(
        pilot_task=None, pilot_worker_private_bytes=None, resolved_workers=0, mode="test"
    )
    assert report.mode == "test"
    assert report.fallback_reason == ""


def test_clear_globals_resets_all() -> None:
    import src.domain.futures.strategy.tiered_workflow.snapshot_executor as se

    se._GLOBAL_EVIDENCE_STORE = MagicMock()
    se._GLOBAL_CFG = MagicMock()
    se._GLOBAL_SYMBOLS = ("BTCUSDT",)
    se._GLOBAL_SEED = 42
    se._GLOBAL_PROBE_DIVERSITY_CORR = {}

    _clear_globals()

    assert se._GLOBAL_EVIDENCE_STORE is None
    assert se._GLOBAL_CFG is None
    assert se._GLOBAL_SYMBOLS is None
    assert se._GLOBAL_SEED == 0
    assert se._GLOBAL_PROBE_DIVERSITY_CORR is None


def test_snapshot_matured_count_empty_df_returns_zero() -> None:
    cfg = MagicMock()
    result = _snapshot_matured_count(pd.DataFrame(), 100, cfg)
    assert result == 0


def test_snapshot_matured_count_no_exit_idx_column_returns_zero() -> None:
    cfg = MagicMock()
    df = pd.DataFrame({"symbol": ["BTCUSDT"]})
    result = _snapshot_matured_count(df, 100, cfg)
    assert result == 0


def test_snapshot_matured_count_counts_matured() -> None:
    cfg = MagicMock()
    cfg.l1_evidence_lookback_bars = None
    df = pd.DataFrame({"exit_idx": [10, 50, 100, 200]})
    result = _snapshot_matured_count(df, 100, cfg)
    assert result == 2


def test_snapshot_matured_count_with_lookback() -> None:
    cfg = MagicMock()
    cfg.l1_evidence_lookback_bars = 50
    df = pd.DataFrame({"exit_idx": [10, 30, 60, 90]})
    result = _snapshot_matured_count(df, 100, cfg)
    assert result == 2


def test_snapshot_matured_count_with_lookback_filters_old() -> None:
    cfg = MagicMock()
    cfg.l1_evidence_lookback_bars = 30
    df = pd.DataFrame({"exit_idx": [10, 50, 80]})
    result = _snapshot_matured_count(df, 100, cfg)
    assert result == 1


def test_execute_snapshot_task_raises_when_globals_unset() -> None:
    with pytest.raises(ValueError, match="snapshot globals not set"):
        _execute_snapshot_task(L1SnapshotTask(0, 100))


@patch("src.domain.futures.strategy.tiered_workflow.signal_selection.compute_symbol_strategy_evidence")
@patch("src.domain.futures.strategy.tiered_workflow.signal_selection.build_qualified_signal_registry")
def test_execute_snapshot_task_with_globals(
    mock_build_registry, mock_compute_evidence
) -> None:
    import src.domain.futures.strategy.tiered_workflow.snapshot_executor as se

    cfg = MagicMock()
    cfg.l1_min_signals_per_symbol = 1
    cfg.l1_evidence_lookback_bars = None

    mock_registry = MagicMock()
    mock_build_registry.return_value = mock_registry
    mock_compute_evidence.return_value = ()

    se._GLOBAL_EVIDENCE_STORE = pd.DataFrame({
        "exit_idx": [10, 50],
        "symbol": ["A", "B"],
        "strategy_id": ["s1", "s2"],
    })
    se._GLOBAL_CFG = cfg
    se._GLOBAL_SYMBOLS = ("BTCUSDT",)
    se._GLOBAL_SEED = 42
    se._GLOBAL_PROBE_DIVERSITY_CORR = None

    result = _execute_snapshot_task(L1SnapshotTask(snapshot_offset=0, as_of_idx=100))
    assert result.as_of_idx == 100
    assert result.matured_event_count == 2


@patch("src.domain.futures.strategy.tiered_workflow.signal_selection.compute_symbol_strategy_evidence")
@patch("src.domain.futures.strategy.tiered_workflow.signal_selection.build_qualified_signal_registry")
def test_execute_snapshot_task_with_lookback_bars(
    mock_build_registry, mock_compute_evidence
) -> None:
    import src.domain.futures.strategy.tiered_workflow.snapshot_executor as se

    cfg = MagicMock()
    cfg.l1_min_signals_per_symbol = 1
    cfg.l1_evidence_lookback_bars = 30

    mock_build_registry.return_value = MagicMock()
    mock_compute_evidence.return_value = ()

    se._GLOBAL_EVIDENCE_STORE = pd.DataFrame({
        "exit_idx": [10, 50, 80, 120],
        "symbol": ["A", "B", "C", "D"],
        "strategy_id": ["s1", "s2", "s3", "s4"],
    }).sort_values("exit_idx").reset_index(drop=True)
    se._GLOBAL_CFG = cfg
    se._GLOBAL_SYMBOLS = ("BTCUSDT",)
    se._GLOBAL_SEED = 42

    result = _execute_snapshot_task(L1SnapshotTask(snapshot_offset=0, as_of_idx=100))
    assert result.as_of_idx == 100


def test_execute_empty_tasks_returns_no_tasks_mode() -> None:
    cfg = MagicMock()
    result, report = execute_l1_snapshot_batch(
        evidence_store=pd.DataFrame(),
        tasks=(),
        cfg=cfg,
        symbols=("BTCUSDT",),
        seed=42,
    )
    assert result == ()
    assert report.mode == "no_tasks"
    assert report.resolved_workers == 0


def test_execute_single_task_serial_few_tasks(monkeypatch) -> None:
    import src.domain.futures.strategy.tiered_workflow.snapshot_executor as se

    monkeypatch.setattr(se, "_execute_snapshot_task", _fake_execute_snapshot_task)

    cfg = MagicMock()
    evidence_store = pd.DataFrame()
    tasks = (L1SnapshotTask(snapshot_offset=0, as_of_idx=100),)

    result, report = execute_l1_snapshot_batch(
        evidence_store=evidence_store,
        tasks=tasks,
        cfg=cfg,
        symbols=("BTCUSDT",),
        seed=42,
    )
    assert len(result) == 1
    assert report.mode == "serial_few_tasks"
    assert report.resolved_workers == 1
    assert result[0].as_of_idx == 100


def test_execute_two_tasks_infrastructure_failure_serial_all(monkeypatch) -> None:
    import src.domain.futures.strategy.tiered_workflow.snapshot_executor as se

    monkeypatch.setattr(se, "_execute_snapshot_task", _fake_execute_snapshot_task)

    class FakeFailingPool:
        def __init__(self, *a, **kw):
            raise OSError("mock fork not available")
        def __enter__(self):
            return self
        def __exit__(self, *a, **kw):
            pass

    monkeypatch.setattr(se, "ProcessPoolExecutor", FakeFailingPool)

    cfg = MagicMock()
    evidence_store = pd.DataFrame()
    tasks = (
        L1SnapshotTask(snapshot_offset=0, as_of_idx=100),
        L1SnapshotTask(snapshot_offset=1, as_of_idx=200),
    )

    result, report = execute_l1_snapshot_batch(
        evidence_store=evidence_store,
        tasks=tasks,
        cfg=cfg,
        symbols=("BTCUSDT",),
        seed=42,
    )
    assert len(result) == 2
    assert report.mode == "serial_all_infrastructure_failure"
    assert se._GLOBAL_EVIDENCE_STORE is None


def test_execute_two_tasks_serial_remaining_no_pss(monkeypatch) -> None:
    import src.domain.futures.strategy.tiered_workflow.snapshot_executor as se

    monkeypatch.setattr(se, "_execute_snapshot_task", _fake_execute_snapshot_task)

    fake_memory = MagicMock(
        tree_pss_bytes=None, tree_uss_bytes=None, parent_rss_bytes=0, available_bytes=0
    )

    class FakePool:
        def __init__(self, *a, **kw):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a, **kw):
            pass
        def submit(self, fn, task):
            fut: Future = Future()
            fut.set_result(fn(task))
            return fut
        def shutdown(self, *a, **kw):
            pass

    monkeypatch.setattr(se, "ProcessPoolExecutor", FakePool)
    monkeypatch.setattr(se, "snapshot_process_tree_memory", lambda pid: fake_memory)

    cfg = MagicMock()
    evidence_store = pd.DataFrame()
    tasks = (
        L1SnapshotTask(snapshot_offset=0, as_of_idx=100),
        L1SnapshotTask(snapshot_offset=1, as_of_idx=200),
    )

    result, report = execute_l1_snapshot_batch(
        evidence_store=evidence_store,
        tasks=tasks,
        cfg=cfg,
        symbols=("BTCUSDT",),
        seed=42,
    )
    assert len(result) == 2
    assert report.mode == "serial_remaining_no_pss"
    assert report.fallback_reason == "pss_unavailable"
    assert se._GLOBAL_EVIDENCE_STORE is None


def test_execute_two_tasks_pilot_measured_parallel_remaining(monkeypatch) -> None:
    import src.domain.futures.strategy.tiered_workflow.snapshot_executor as se

    monkeypatch.setattr(se, "_execute_snapshot_task", _fake_execute_snapshot_task)

    fake_memory = MagicMock(
        tree_pss_bytes=100_000_000,
        tree_uss_bytes=50_000_000,
        parent_rss_bytes=200_000_000,
        available_bytes=10_000_000_000,
    )
    fake_plan = MagicMock(workers=2, binding_constraint="cpu_cap", projected_tree_bytes=200_000_000, reason="ok")
    called_plan = False

    def fake_resolve(**kw):
        nonlocal called_plan
        called_plan = True
        return fake_plan

    class FakePool:
        def __init__(self, *a, **kw):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a, **kw):
            pass
        def submit(self, fn, task):
            fut: Future = Future()
            fut.set_result(fn(task))
            return fut
        def shutdown(self, *a, **kw):
            pass

    memory_calls = [0]

    def fake_memory_fn(pid: int) -> MagicMock:
        memory_calls[0] += 1
        if memory_calls[0] == 1:
            return MagicMock(
                tree_pss_bytes=50_000_000,
                tree_uss_bytes=25_000_000,
                parent_rss_bytes=100_000_000,
                available_bytes=10_000_000_000,
            )
        return MagicMock(
            tree_pss_bytes=100_000_000,
            tree_uss_bytes=50_000_000,
            parent_rss_bytes=200_000_000,
            available_bytes=10_000_000_000,
        )

    monkeypatch.setattr(se, "ProcessPoolExecutor", FakePool)
    monkeypatch.setattr(se, "snapshot_process_tree_memory", fake_memory_fn)
    monkeypatch.setattr(se, "resolve_post_pilot_memory_plan", fake_resolve)

    cfg = MagicMock()
    evidence_store = pd.DataFrame()
    tasks = (
        L1SnapshotTask(snapshot_offset=0, as_of_idx=100),
        L1SnapshotTask(snapshot_offset=1, as_of_idx=200),
    )

    result, report = execute_l1_snapshot_batch(
        evidence_store=evidence_store,
        tasks=tasks,
        cfg=cfg,
        symbols=("BTCUSDT",),
        seed=42,
    )
    assert len(result) == 2
    assert called_plan
    assert se._GLOBAL_EVIDENCE_STORE is None
