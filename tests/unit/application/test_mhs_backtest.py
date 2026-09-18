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
import src.mhs.backtest.contracts as bt_contracts
import src.mhs.backtest.inventory as bt_inventory
from src.mhs.backtest.contracts import ProcessInventoryBacktestError
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
        "result_output": tmp_path / "result.json",
    }
    args.update(overrides)
    return MhsBacktestRequest(**args)


def _completed_report():
    from src.mhs.resources import ProcessTreeMemoryStats

    targets = _inventory_test_targets()
    proxy = _inventory_test_proxy(targets)
    grid = pd.date_range("2022-01-01", periods=960, freq="3min", tz="UTC")
    equity = pd.Series(1.0, index=grid)
    base = _inventory_fake_result(equity, valid=True)
    stress = _inventory_fake_result(equity, valid=True)
    gate = bt_inventory._inventory_gate(base, stress)
    stats = ProcessTreeMemoryStats(
        tree_pss_peak_bytes=10,
        tree_uss_peak_bytes=9,
        min_system_available_bytes=8,
        max_concurrent_procs=1,
        samples_taken=2,
    )
    return bt_contracts.ProcessInventoryReport(
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
        _request(tmp_path, result_output=tmp_path / "primary.csv"),
        _request(tmp_path, targets_output=tmp_path / "targets.txt"),
    ]
    for request in bad:
        with pytest.raises(ValueError, match=r".+"):
            execute_mhs_backtest(request)
    with pytest.raises(ValueError, match="MhsBacktestRequest"):
        validate_mhs_backtest_request("not-a-request")
    assert list(tmp_path.iterdir()) == []


def test_validate_mhs_backtest_request_rejects_occupied_destinations(tmp_path, monkeypatch) -> None:
    """Fresh distinct evidence: aliased, existing or symlinked destinations are rejected."""
    import src.application.mhs_backtest as svc

    def _boom(*args, **kwargs):
        raise AssertionError("evaluation must not run")

    monkeypatch.setattr(svc, "evaluate_process_inventory_backtest", _boom)
    start, end = _stamps()
    alias_targets = tmp_path / "alias.parquet"
    os.symlink(tmp_path / "result.json", alias_targets)
    with pytest.raises(ValueError, match="distinct"):
        validate_mhs_backtest_request(
            _request(tmp_path, targets_output=alias_targets)
        )
    occupied = _request(tmp_path)
    occupied.result_output.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="fresh"):
        validate_mhs_backtest_request(occupied)
    assert json.loads(occupied.result_output.read_text(encoding="utf-8")) == {}
    link = tmp_path / "link.json"
    os.symlink(tmp_path / "missing-target.json", link)
    with pytest.raises(ValueError, match="fresh"):
        validate_mhs_backtest_request(_request(tmp_path, result_output=link))
    assert {p.name for p in tmp_path.iterdir()} == {
        occupied.result_output.name, link.name, alias_targets.name,
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
    payload = json.loads(request.result_output.read_text(encoding="utf-8"))
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

    cause = ValueError("evaluator blew up")
    failure = bt_contracts.ProcessInventoryBacktestError(
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
    payload = json.loads(request.result_output.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["error_code"] == "MEMORY_BUDGET"
    assert payload["stage"] == "process_execution_piece"


def test_execute_mhs_backtest_reports_failure_persist_error(tmp_path, monkeypatch) -> None:
    """Persistence failure: a failure-report write error never replaces the original evaluation error."""
    import src.application.mhs_backtest as svc

    failure = bt_contracts.ProcessInventoryBacktestError(_failure_report_fixture())

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
    assert not request.result_output.exists()


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
    payload = json.loads(request.result_output.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert not request.targets_output.exists()


def _worker_args(tmp_path: Path, **overrides):
    args = {
        "start": "2022-01-01T00:00:00+00:00",
        "end": "2022-01-04T00:00:00+00:00",
        "result-output": str(tmp_path / "result.json"),
        "total_tree_pss_bytes": 4 * 2**30,
        "replay_tree_pss_bytes": 3 * 2**30,
        "min_available_bytes": 2 * 2**30,
    }
    args.update(overrides)
    argv = [
        "--start", str(args.pop("start")),
        "--end", str(args.pop("end")),
        "--result-output", str(args.pop("result-output")),
        "--total-tree-pss-bytes", str(args.pop("total_tree_pss_bytes")),
        "--replay-tree-pss-bytes", str(args.pop("replay_tree_pss_bytes")),
        "--min-available-bytes", str(args.pop("min_available_bytes")),
    ]
    for key, value in args.items():
        argv += [f"--{key}", str(value)]
    return argv


def _managed_registry(tmp_path: Path, run_id: str):
    from src.backtests.contracts import RunRegistration
    from src.backtests.registry import initialize_registry, register_run

    db = tmp_path / "registry.sqlite3"
    initialize_registry(db)
    register_run(
        db,
        RunRegistration(
            run_id=run_id, strategy_id="strat-a", registered_at="2026-01-01T00:00:00+00:00",
            request={"window": "3m"}, managed_directory=None,
        ),
    )
    return db


def test_validate_managed_context_requires_registered_run(tmp_path) -> None:
    """Managed context: partial ownership, bad identities and unknown runs fail validation."""
    import uuid

    from src.application.mhs_backtest import validate_mhs_backtest_request

    run_id = uuid.uuid4().hex
    db = _managed_registry(tmp_path, run_id)
    evidence_root = tmp_path / "evidence"
    good = _request(
        tmp_path, evidence_root=evidence_root, registry_path=db, run_id=run_id,
    )
    validate_mhs_backtest_request(good)
    with pytest.raises(ValueError, match=r".+"):
        validate_mhs_backtest_request(_request(tmp_path, registry_path=db, run_id=run_id))
    with pytest.raises(ValueError, match=r".+"):
        validate_mhs_backtest_request(_request(tmp_path, run_id=run_id))
    with pytest.raises(ValueError, match=r".+"):
        validate_mhs_backtest_request(
            _request(tmp_path, evidence_root="root", registry_path=db, run_id=run_id)
        )
    with pytest.raises(ValueError, match=r".+"):
        validate_mhs_backtest_request(
            _request(tmp_path, evidence_root=evidence_root, registry_path="db", run_id=run_id)
        )
    with pytest.raises(ValueError, match=r".+"):
        validate_mhs_backtest_request(
            _request(tmp_path, evidence_root=evidence_root, registry_path=db, run_id="bogus")
        )
    with pytest.raises(ValueError, match=r".+"):
        validate_mhs_backtest_request(
            _request(
                tmp_path, evidence_root=evidence_root, registry_path=db,
                run_id=uuid.uuid4().hex,
            )
        )
    with pytest.raises(ValueError, match=r".+"):
        validate_mhs_backtest_request(
            _request(
                tmp_path, evidence_root=evidence_root,
                registry_path=tmp_path / "absent.sqlite3", run_id=run_id,
            )
        )
    empty = tmp_path / "empty.sqlite3"
    empty.write_bytes(b"not a database")
    with pytest.raises(ValueError, match=r".+"):
        validate_mhs_backtest_request(
            _request(tmp_path, evidence_root=evidence_root, registry_path=empty, run_id=run_id)
        )


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
    from src.application import mhs_worker as worker

    failure = bt_contracts.ProcessInventoryBacktestError(_failure_report_fixture())

    def _boom(request):
        raise failure

    monkeypatch.setattr(wmod, "execute_mhs_backtest", _boom)
    with pytest.raises(bt_contracts.ProcessInventoryBacktestError) as excinfo:
        worker.main(_worker_args(tmp_path))
    assert excinfo.value is failure


def test_worker_main_logs_start_and_persistence_completion(tmp_path, monkeypatch, caplog) -> None:
    """Worker diagnostics reach standard logging without touching the CLI configuration."""
    import logging

    import src.application.mhs_worker as wmod
    from src.application import mhs_worker as worker

    source = __import__("pathlib").Path(wmod.__file__).read_text(encoding="utf-8")
    assert "src.cli" not in source
    monkeypatch.setattr(wmod, "execute_mhs_backtest", lambda request: object())
    with caplog.at_level(logging.INFO, logger="src.application.mhs_worker"):
        assert worker.main(_worker_args(tmp_path)) == 0
    messages = [r.getMessage() for r in caplog.records if r.name == "src.application.mhs_worker"]
    assert any("status=start" in m and "2022-01-01" in m for m in messages)
    assert any("status=persistence_complete" in m and "result.json" in m for m in messages)


def test_worker_main_preserves_existing_root_handler(tmp_path, monkeypatch) -> None:
    """Repeated in-process worker calls neither reset nor duplicate user-installed handlers."""
    import logging

    import src.application.mhs_worker as wmod
    from src.application import mhs_worker as worker

    root = logging.getLogger()
    saved = list(root.handlers)
    marker: list[str] = []
    custom = logging.Handler()
    custom.emit = lambda record: marker.append(record.getMessage())  # type: ignore[method-assign]
    root.handlers = [custom]
    try:
        monkeypatch.setattr(wmod, "execute_mhs_backtest", lambda request: object())
        assert worker.main(_worker_args(tmp_path)) == 0
        assert worker.main(_worker_args(tmp_path)) == 0
        assert root.handlers == [custom]
    finally:
        root.handlers = saved


def test_worker_main_validation_failure_without_result(tmp_path, monkeypatch, caplog) -> None:
    """Invalid worker controls keep existing classification and fabricate no result."""
    import logging

    import src.application.mhs_worker as wmod
    from src.application import mhs_worker as worker

    def _boom(request):
        raise AssertionError("evaluation must not run")

    monkeypatch.setattr(wmod, "execute_mhs_backtest", _boom)
    with caplog.at_level(logging.INFO, logger="src.application.mhs_worker"):
        with pytest.raises(SystemExit, match="invalid worker arguments"):
            worker.main(_worker_args(tmp_path, start="2022-01-01T00:00:00"))
        with pytest.raises(SystemExit, match="invalid worker arguments"):
            worker.main(
                _worker_args(
                    tmp_path, total_tree_pss_bytes=1 * 2**30, replay_tree_pss_bytes=2 * 2**30,
                )
            )
    messages = [r.getMessage() for r in caplog.records if r.name == "src.application.mhs_worker"]
    assert not any("status=persistence_complete" in m for m in messages)


def test_worker_main_failure_propagates_without_completion(tmp_path, monkeypatch, caplog) -> None:
    """Typed inventory failure keeps dedicated persistence and emits no completion event."""
    import logging

    import src.application.mhs_worker as wmod
    from src.application import mhs_worker as worker

    failure = bt_contracts.ProcessInventoryBacktestError(_failure_report_fixture())

    def _boom(request):
        raise failure

    monkeypatch.setattr(wmod, "execute_mhs_backtest", _boom)
    with caplog.at_level(logging.INFO, logger="src.application.mhs_worker"), pytest.raises(
        bt_contracts.ProcessInventoryBacktestError
    ) as excinfo:
        worker.main(_worker_args(tmp_path))
    assert excinfo.value is failure
    messages = [r.getMessage() for r in caplog.records if r.name == "src.application.mhs_worker"]
    assert any("status=start" in m for m in messages)
    assert not any("status=persistence_complete" in m for m in messages)


def test_worker_main_completion_claims_no_certification(tmp_path, monkeypatch, caplog) -> None:
    """Completion on a financially invalid report claims persistence only, never certification."""
    import logging

    import src.application.mhs_worker as wmod
    from src.application import mhs_worker as worker

    monkeypatch.setattr(wmod, "execute_mhs_backtest", lambda request: object())
    with caplog.at_level(logging.INFO, logger="src.application.mhs_worker"):
        assert worker.main(_worker_args(tmp_path)) == 0
    messages = [r.getMessage() for r in caplog.records if r.name == "src.application.mhs_worker"]
    completion = [m for m in messages if "status=persistence_complete" in m]
    assert len(completion) == 1
    lowered = completion[0].lower()
    assert "primary_valid" not in lowered
    assert "gate" not in lowered
    assert "deploy" not in lowered
    assert "certif" not in lowered


def test_worker_main_forwards_managed_context(tmp_path, monkeypatch) -> None:
    """Worker managed context: internal publication arguments reach the service request."""
    import uuid

    import src.application.mhs_worker as wmod
    from src.application import mhs_worker as worker

    seen: dict = {}

    def _spy(request):
        seen["request"] = request
        return object()

    monkeypatch.setattr(wmod, "execute_mhs_backtest", _spy)
    run_id = uuid.uuid4().hex
    evidence_root = tmp_path / "evidence"
    db = _managed_registry(tmp_path, run_id)
    assert worker.main(
        _worker_args(
            tmp_path, **{
                "evidence-root": str(evidence_root),
                "registry-path": str(db),
                "run-id": run_id,
            }
        )
    ) == 0
    request = seen["request"]
    assert request.evidence_root == evidence_root
    assert request.registry_path == db
    assert request.run_id == run_id


def test_worker_main_defaults_to_standalone_context(tmp_path, monkeypatch) -> None:
    """Worker standalone context: omitted publication arguments keep the standalone report contract."""
    import src.application.mhs_worker as wmod
    from src.application import mhs_worker as worker

    seen: dict = {}

    def _spy(request):
        seen["request"] = request
        return object()

    monkeypatch.setattr(wmod, "execute_mhs_backtest", _spy)
    assert worker.main(_worker_args(tmp_path)) == 0
    request = seen["request"]
    assert request.evidence_root is None
    assert request.registry_path is None
    assert request.run_id is None
