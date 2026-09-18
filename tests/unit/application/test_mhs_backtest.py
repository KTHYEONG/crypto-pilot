"""Invariant scenarios for the source-owned backtest service and worker."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from src.application.mhs_backtest import (
    MhsBacktestRequest,
    execute_mhs_backtest,
    validate_mhs_backtest_request,
)
from src.mhs.params import PROCESS_EVALUATION_CEILING
from src.mhs.process_backtest import ProcessInventoryBacktestError
from tests.unit.mhs.test_process_backtest import (
    _failure_report_fixture,
    _inventory_fake_result,
    _inventory_test_proxy,
    _inventory_test_targets,
)


def _stamps():
    return pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-01-04", tz="UTC")


def _request(tmp_path: Path, **overrides):
    start, end = _stamps()
    args: dict = {
        "start": start,
        "end": end,
        "data_root": None,
        "output": tmp_path / "primary.json",
        "failure_output": tmp_path / "failure.json",
    }
    args.update(overrides)
    return MhsBacktestRequest(**args)


def _completed_report():
    import src.mhs.process_backtest as pb
    from src.mhs.resources import ProcessTreeMemoryStats

    targets = _inventory_test_targets()
    proxy = _inventory_test_proxy(targets)
    grid = pd.date_range("2022-01-01", periods=960, freq="3min", tz="UTC")
    equity = pd.Series(1.0, index=grid)
    base = _inventory_fake_result(equity, valid=True)
    stress = _inventory_fake_result(equity, valid=True)
    gate = pb._inventory_gate(base, stress)
    stats = ProcessTreeMemoryStats(
        tree_pss_peak_bytes=10,
        tree_uss_peak_bytes=9,
        min_system_available_bytes=8,
        max_concurrent_procs=1,
        samples_taken=2,
    )
    return pb.ProcessInventoryReport(
        proxy=proxy,
        base=base,
        stress=stress,
        gate=gate,
        resource_measurements=(),
        memory_stats=stats,
    )


def test_validate_mhs_backtest_request_rejects_bad_boundaries(tmp_path, monkeypatch) -> None:
    """Request boundary: invalid dates or policy are rejected before evaluation and output writes."""
    import src.application.mhs_backtest as svc

    def _boom(*args, **kwargs):
        raise AssertionError("evaluation must not run")

    monkeypatch.setattr(svc, "evaluate_process_inventory_backtest", _boom)
    start, end = _stamps()
    bad = [
        _request(tmp_path, start=pd.Timestamp("2022-01-01")),
        _request(tmp_path, end=pd.Timestamp("2022-01-04")),
        _request(tmp_path, start=end, end=start),
        _request(tmp_path, end=PROCESS_EVALUATION_CEILING + pd.Timedelta(seconds=1)),
        _request(tmp_path, tracking_error_threshold=-1.0),
        _request(tmp_path, tracking_error_threshold=float("nan")),
        _request(tmp_path, tracking_error_threshold="bad"),
        _request(tmp_path, output=tmp_path / "primary.csv"),
        _request(tmp_path, targets_output=tmp_path / "targets.txt"),
    ]
    for request in bad:
        with pytest.raises(ValueError, match=r".+"):
            execute_mhs_backtest(request)
    with pytest.raises(ValueError, match="MhsBacktestRequest"):
        validate_mhs_backtest_request("not-a-request")
    assert list(tmp_path.iterdir()) == []


def test_validate_mhs_backtest_request_rejects_occupied_destinations(tmp_path, monkeypatch) -> None:
    """Fresh distinct evidence: aliased, reserved, existing or symlinked destinations are rejected."""
    import src.application.mhs_backtest as svc
    from src.mhs.process_backtest import PROCESS_REPORT_PATH

    def _boom(*args, **kwargs):
        raise AssertionError("evaluation must not run")

    monkeypatch.setattr(svc, "evaluate_process_inventory_backtest", _boom)
    start, end = _stamps()
    with pytest.raises(ValueError, match="distinct"):
        validate_mhs_backtest_request(_request(tmp_path, failure_output=tmp_path / "primary.json"))
    alias_targets = tmp_path / "alias.parquet"
    os.symlink(tmp_path / "primary.json", alias_targets)
    with pytest.raises(ValueError, match="distinct"):
        validate_mhs_backtest_request(
            _request(tmp_path, targets_output=alias_targets)
        )
    with pytest.raises(ValueError, match="reserved"):
        validate_mhs_backtest_request(_request(tmp_path, output=PROCESS_REPORT_PATH))
    alias = tmp_path / "alias.json"
    os.symlink(tmp_path / "primary.json", alias)
    with pytest.raises(ValueError, match="distinct"):
        validate_mhs_backtest_request(
            _request(tmp_path, failure_output=alias)
        )
    occupied = _request(tmp_path)
    occupied.output.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="fresh"):
        validate_mhs_backtest_request(occupied)
    assert json.loads(occupied.output.read_text(encoding="utf-8")) == {}
    link = tmp_path / "link.json"
    os.symlink(tmp_path / "missing-target.json", link)
    with pytest.raises(ValueError, match="fresh"):
        validate_mhs_backtest_request(_request(tmp_path, output=link))
    assert {p.name for p in tmp_path.iterdir()} == {
        occupied.output.name, link.name, alias.name, alias_targets.name,
    }


def test_execute_mhs_backtest_persists_primary_inventory_evidence(tmp_path, monkeypatch) -> None:
    """Primary inventory evidence: the existing 3m schema is persisted with comparative proxy."""
    import src.application.mhs_backtest as svc
    from src.mhs.resources import MhsMemoryBudget, resolve_mhs_memory_budget

    report = _completed_report()
    seen: dict = {}

    def _spy(*args, **kwargs):
        seen.update(kwargs)
        return report

    monkeypatch.setattr(svc, "evaluate_process_inventory_backtest", _spy)
    explicit = MhsMemoryBudget()
    request = _request(tmp_path, memory_budget=explicit)
    got = execute_mhs_backtest(request)
    assert got is report
    assert seen["execution_policy"].tracking_error_threshold is None
    assert seen["memory_budget"] == resolve_mhs_memory_budget(explicit)
    payload = json.loads(request.output.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["execution_timeframe"] == "3m"
    assert payload["certification_level"] == "process_inventory_3m"
    assert payload["proxy"]["scope"] == "hourly_proxy_comparison"


def test_execute_mhs_backtest_exports_exact_sized_targets(tmp_path, monkeypatch) -> None:
    """Exact targets: export uses exact sized base targets, not rebuilt or unit targets."""
    import src.application.mhs_backtest as svc

    report = _completed_report()
    monkeypatch.setattr(svc, "evaluate_process_inventory_backtest", lambda *a, **k: report)
    seen: dict = {}
    real_targets = svc.persist_process_targets

    def _spy(path, output):
        seen["path"] = path
        seen["output"] = output
        return real_targets(path, output)

    monkeypatch.setattr(svc, "persist_process_targets", _spy)
    request = _request(tmp_path, targets_output=tmp_path / "targets.parquet")
    execute_mhs_backtest(request)
    assert seen["path"] is report.proxy.base
    assert seen["output"] == request.targets_output
    back = pd.read_parquet(request.targets_output)
    pd.testing.assert_frame_equal(
        back, report.proxy.base.target_weights.astype("float64"), check_freq=False
    )


def test_execute_mhs_backtest_preserves_typed_failure(tmp_path, monkeypatch) -> None:
    """Domain failure preservation: dedicated failure evidence is persisted and the cause escapes intact."""
    import src.application.mhs_backtest as svc
    import src.mhs.process_backtest as pb

    cause = ValueError("evaluator blew up")
    failure = pb.ProcessInventoryBacktestError(
        _failure_report_fixture(error_code="MEMORY_BUDGET", stage="process_execution_piece")
    )

    def _boom(*args, **kwargs):
        raise failure from cause

    monkeypatch.setattr(svc, "evaluate_process_inventory_backtest", _boom)
    request = _request(tmp_path)
    with pytest.raises(ProcessInventoryBacktestError) as excinfo:
        execute_mhs_backtest(request)
    assert excinfo.value is failure
    assert excinfo.value.__cause__ is cause
    payload = json.loads(request.failure_output.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["error_code"] == "MEMORY_BUDGET"
    assert payload["stage"] == "process_execution_piece"
    assert not request.output.exists()


def test_execute_mhs_backtest_reports_failure_persist_error(tmp_path, monkeypatch) -> None:
    """Persistence failure: a failure-report write error never replaces the original evaluation error."""
    import src.application.mhs_backtest as svc
    import src.mhs.process_backtest as pb

    failure = pb.ProcessInventoryBacktestError(_failure_report_fixture())

    def _boom(*args, **kwargs):
        raise failure

    def _persist_boom(report, output):
        raise OSError("disk gone")

    monkeypatch.setattr(svc, "evaluate_process_inventory_backtest", _boom)
    monkeypatch.setattr(svc, "persist_process_inventory_failure", _persist_boom)
    request = _request(tmp_path)
    with pytest.raises(ProcessInventoryBacktestError) as excinfo:
        execute_mhs_backtest(request)
    assert excinfo.value is failure
    assert not request.output.exists()
    assert not request.failure_output.exists()


def test_execute_mhs_backtest_reports_target_export_error(tmp_path, monkeypatch) -> None:
    """Persistence failure: a target-export write error stays failed while keeping primary evidence."""
    import src.application.mhs_backtest as svc

    report = _completed_report()
    monkeypatch.setattr(svc, "evaluate_process_inventory_backtest", lambda *a, **k: report)

    def _target_boom(path, output):
        raise OSError("targets unwritable")

    monkeypatch.setattr(svc, "persist_process_targets", _target_boom)
    request = _request(tmp_path, targets_output=tmp_path / "targets.parquet")
    with pytest.raises(OSError, match="targets unwritable"):
        execute_mhs_backtest(request)
    payload = json.loads(request.output.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert not request.targets_output.exists()


def _worker_args(tmp_path: Path, **overrides):
    args = {
        "start": "2022-01-01T00:00:00+00:00",
        "end": "2022-01-04T00:00:00+00:00",
        "output": str(tmp_path / "primary.json"),
        "failure_output": str(tmp_path / "failure.json"),
        "total_tree_pss_bytes": 4 * 2**30,
        "replay_tree_pss_bytes": 3 * 2**30,
        "min_available_bytes": 2 * 2**30,
    }
    args.update(overrides)
    argv = [
        "--start", str(args.pop("start")),
        "--end", str(args.pop("end")),
        "--output", str(args.pop("output")),
        "--failure-output", str(args.pop("failure_output")),
        "--total-tree-pss-bytes", str(args.pop("total_tree_pss_bytes")),
        "--replay-tree-pss-bytes", str(args.pop("replay_tree_pss_bytes")),
        "--min-available-bytes", str(args.pop("min_available_bytes")),
    ]
    for key, value in args.items():
        argv += [f"--{key}", str(value)]
    return argv


def test_worker_main_rejects_invalid_arguments(tmp_path) -> None:
    """Worker argument validation: bad timestamps or budgets exit without evaluating."""
    from src.application import mhs_worker as worker

    with pytest.raises(SystemExit):
        worker.main(_worker_args(tmp_path, start="not-a-date"))
    with pytest.raises(SystemExit):
        worker.main(_worker_args(tmp_path, start="2022-01-01T00:00:00"))
    with pytest.raises(SystemExit):
        worker.main(
            _worker_args(
                tmp_path, total_tree_pss_bytes=1 * 2**30, replay_tree_pss_bytes=2 * 2**30,
            )
        )
    with pytest.raises(SystemExit):
        worker.main(["--start", "2022-01-01T00:00:00+00:00"])


def test_worker_main_executes_service_request(tmp_path, monkeypatch) -> None:
    """Worker dispatch: one request is built and executed exactly once."""
    import src.application.mhs_worker as wmod
    from src.application import mhs_worker as worker
    from src.application.mhs_backtest import MhsBacktestRequest

    seen: dict = {}

    def _spy(request):
        seen["request"] = request
        return object()

    monkeypatch.setattr(wmod, "execute_mhs_backtest", _spy)
    code = worker.main(
        _worker_args(
            tmp_path,
            start="2022-01-01T00:00:00+09:00",
            **{"rebalance-tracking-error-threshold": "0.2"},
        )
    )
    assert code == 0
    request = seen["request"]
    assert isinstance(request, MhsBacktestRequest)
    assert request.tracking_error_threshold == 0.2
    assert request.memory_budget is not None
    assert request.memory_budget.total_tree_pss_bytes == 4 * 2**30
    assert request.memory_budget.replay_tree_pss_bytes == 3 * 2**30
    assert request.memory_budget.min_available_bytes == 2 * 2**30


def test_worker_main_propagates_evaluation_failure(tmp_path, monkeypatch) -> None:
    """Worker failure: evaluation errors escape nonzero with diagnostics preserved."""
    import src.application.mhs_worker as wmod
    import src.mhs.process_backtest as pb
    from src.application import mhs_worker as worker

    failure = pb.ProcessInventoryBacktestError(_failure_report_fixture())

    def _boom(request):
        raise failure

    monkeypatch.setattr(wmod, "execute_mhs_backtest", _boom)
    with pytest.raises(pb.ProcessInventoryBacktestError) as excinfo:
        worker.main(_worker_args(tmp_path))
    assert excinfo.value is failure
